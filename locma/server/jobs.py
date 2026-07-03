"""Background job runner for the web panel's experiments.

A *job* is a list of picklable cells — ``(fn, args)`` with a top-level ``fn`` —
plus a reduce step over the collected cell results. Cells fan out over a
shared ``ProcessPoolExecutor`` (the same pattern as ``ceiling-eval
--workers``); with ``workers <= 1`` they run inline on the job's collector
thread instead (no pool spin-up — also the test path). The collector updates
``progress`` as cells finish, reduces at the end, and persists the finished
job as JSON so results survive a server restart.

Live visualization has two channels:

- ``on_cell(job, index, result)`` — called on the collector thread as each
  cell finishes; kind-specific closures append to ``job.series`` (named
  ``[x, y]`` point lists) or ``job.live`` (arbitrary partial state, e.g. a
  league matrix filling in).
- ``tail`` — for long single cells (training) that cannot report per-cell:
  the cell appends JSONL lines to a file and a tailer thread follows it,
  turning numeric fields into series points and the ``x`` field into
  progress. File-based, so it crosses the process boundary with no IPC.

Each job owns a scratch directory (``results_dir/<job_id>/``) for aux files:
metrics stream, ``log.txt`` (error tracebacks land there too), checkpoints.
``cancel_file`` supports cooperative cancel inside a running cell — cancel()
touches it, the cell polls it (SB3 callbacks return False on sight). A cancel
that arrives after every cell already completed still reduces to ``done``.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field

Cell = tuple[Callable, tuple]


@dataclass
class TailConfig:
    """Follow a JSONL metrics file: ``x`` names the progress field; every other
    numeric field becomes a series."""

    path: str
    x: str
    total: int | None = None  # progress_total in x units


@dataclass
class Job:
    job_id: str
    kind: str
    name: str
    params: dict
    state: str = "queued"  # queued | running | done | error | cancelled
    progress_done: int = 0
    progress_total: int = 0
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    series: dict[str, list] = field(default_factory=dict)  # name -> [[x, y], ...]
    live: dict = field(default_factory=dict)  # partial structured state (e.g. matrix)

    def add_point(self, name: str, x: float, y: float) -> None:
        self.series.setdefault(name, []).append([float(x), float(y)])

    def to_dict(self, include_series: bool = True) -> dict:
        d = asdict(self)
        if not include_series:
            d.pop("series", None)
            d.pop("live", None)
        return d


def make_job_id(kind: str) -> str:
    return f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class JobRunner:
    """Runs jobs on daemon threads; cells optionally fan out to a process pool."""

    def __init__(self, results_dir: str | None = None, workers: int = 0):
        self._jobs: dict[str, Job] = {}
        self._cancel: set[str] = set()
        self._cancel_files: dict[str, str] = {}
        self._lock = threading.Lock()
        self._pool: ProcessPoolExecutor | None = None
        self._workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
        self._results_dir = results_dir

    # -- submission / bookkeeping ------------------------------------------

    def job_dir(self, job_id: str) -> str:
        """Per-job scratch directory (metrics stream, logs, checkpoints)."""
        base = self._results_dir or "runs/experiments"
        path = os.path.join(base, job_id)
        os.makedirs(path, exist_ok=True)
        return path

    def submit(
        self,
        *,
        kind: str,
        name: str,
        params: dict,
        cells: list[Cell],
        reduce_fn: Callable[[dict, list], dict],
        on_cell: Callable[[Job, int, object], None] | None = None,
        tail: TailConfig | None = None,
        cancel_file: str | None = None,
        job_id: str | None = None,
    ) -> Job:
        job = Job(
            job_id=job_id or make_job_id(kind),
            kind=kind,
            name=name,
            params=params,
            progress_total=tail.total if tail and tail.total else len(cells),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            if cancel_file:
                self._cancel_files[job.job_id] = cancel_file
        threading.Thread(
            target=self._run, args=(job, cells, reduce_fn, on_cell, tail), daemon=True
        ).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._cancel.add(job_id)
            cancel_file = self._cancel_files.get(job_id)
        if cancel_file:
            try:
                with open(cancel_file, "w", encoding="utf-8") as f:
                    f.write("cancel\n")
            except OSError:
                pass
        return True

    def _cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel

    # -- execution ----------------------------------------------------------

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

            self._pool = ProcessPoolExecutor(
                max_workers=self._workers, initializer=init_eval_worker
            )
        return self._pool

    def _run(self, job: Job, cells: list[Cell], reduce_fn, on_cell, tail) -> None:
        job.state = "running"
        job.started = time.time()
        results: list = [None] * len(cells)
        stop_tail = threading.Event()
        tailer: threading.Thread | None = None
        if tail is not None:
            tailer = threading.Thread(target=self._tail, args=(job, tail, stop_tail), daemon=True)
            tailer.start()

        def collected(i: int, res) -> None:
            results[i] = res
            if tail is None:
                job.progress_done += 1
            if on_cell is not None:
                on_cell(job, i, res)

        try:
            if self._workers <= 1:
                for i, (fn, args) in enumerate(cells):
                    # cancel skips REMAINING cells; a cancel absorbed by the
                    # final cell itself (e.g. training stopped via cancel_file)
                    # still reduces to done, matching the pool path.
                    if i > 0 and self._cancelled(job.job_id):
                        job.state = "cancelled"
                        break
                    collected(i, fn(*args))
            else:
                pool = self._get_pool()
                pending = {pool.submit(fn, *args): i for i, (fn, args) in enumerate(cells)}
                while pending:
                    done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for fut in done:
                        collected(pending.pop(fut), fut.result())
                    if self._cancelled(job.job_id) and pending:
                        for fut in pending:
                            fut.cancel()
                        job.state = "cancelled"
                        break
            if job.state != "cancelled":
                job.result = reduce_fn(job.params, results)
                job.state = "done"
        except Exception as e:  # noqa: BLE001 — job errors are reported, not raised
            job.state = "error"
            job.error = f"{type(e).__name__}: {e}"
            self._write_log(job, traceback.format_exc())
        finally:
            job.finished = time.time()
            if tailer is not None:
                stop_tail.set()
                tailer.join(timeout=5)
            self._persist(job)

    def _tail(self, job: Job, tail: TailConfig, stop: threading.Event) -> None:
        pos = 0

        def drain() -> None:
            nonlocal pos
            try:
                with open(tail.path, encoding="utf-8") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except OSError:
                return
            for line in chunk.splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                x = rec.get(tail.x)
                if not isinstance(x, int | float):
                    continue
                for k, v in rec.items():
                    if k != tail.x and isinstance(v, int | float):
                        job.add_point(k, x, v)
                job.progress_done = int(x)

        while not stop.is_set():
            stop.wait(0.5)
            drain()
        drain()  # final drain after the cell finished

    # -- persistence ---------------------------------------------------------

    def _write_log(self, job: Job, text: str) -> None:
        try:
            path = os.path.join(self.job_dir(job.job_id), "log.txt")
            with open(path, "a", encoding="utf-8") as f:
                f.write(text if text.endswith("\n") else text + "\n")
        except OSError:
            pass

    def log_text(self, job_id: str) -> str | None:
        """Contents of the job's log.txt, or None when it has none."""
        base = self._results_dir or "runs/experiments"
        path = os.path.join(base, job_id, "log.txt")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    def _persist(self, job: Job) -> None:
        if not self._results_dir:
            return
        os.makedirs(self._results_dir, exist_ok=True)
        path = os.path.join(self._results_dir, f"{job.job_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(job.to_dict(), f, indent=2)

    def persisted(self, job_id: str) -> dict | None:
        if not self._results_dir:
            return None
        path = os.path.join(self._results_dir, f"{job_id}.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def history(self) -> list[dict]:
        """Persisted finished jobs (older sessions), excluding in-memory ones.

        Series/live are stripped here to keep the list light; the series
        endpoint re-reads the persisted file on demand."""
        if not self._results_dir or not os.path.isdir(self._results_dir):
            return []
        with self._lock:
            live = set(self._jobs)
        out = []
        for fname in sorted(os.listdir(self._results_dir)):
            if not fname.endswith(".json") or fname[: -len(".json")] in live:
                continue
            try:
                with open(os.path.join(self._results_dir, fname), encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            doc.pop("series", None)
            doc.pop("live", None)
            out.append(doc)
        return out

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
