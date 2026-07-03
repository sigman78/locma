"""Background job runner for the web panel's experiments.

A *job* is a list of picklable cells — ``(fn, args)`` with a top-level ``fn`` —
plus a reduce step over the collected cell results. Cells fan out over a
shared ``ProcessPoolExecutor`` (the same pattern as ``ceiling-eval
--workers``); with ``workers <= 1`` they run inline on the job's collector
thread instead (no pool spin-up — also the test path). The collector updates
``progress`` as cells finish, reduces at the end, and persists the finished
job as JSON so results survive a server restart.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field

Cell = tuple[Callable, tuple]


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

    def to_dict(self) -> dict:
        return asdict(self)


class JobRunner:
    """Runs jobs on daemon threads; cells optionally fan out to a process pool."""

    def __init__(self, results_dir: str | None = None, workers: int = 0):
        self._jobs: dict[str, Job] = {}
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self._pool: ProcessPoolExecutor | None = None
        self._workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
        self._results_dir = results_dir

    # -- submission / bookkeeping ------------------------------------------

    def submit(
        self,
        *,
        kind: str,
        name: str,
        params: dict,
        cells: list[Cell],
        reduce_fn: Callable[[dict, list], dict],
    ) -> Job:
        job = Job(
            job_id=f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}",
            kind=kind,
            name=name,
            params=params,
            progress_total=len(cells),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        threading.Thread(target=self._run, args=(job, cells, reduce_fn), daemon=True).start()
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

    def _run(self, job: Job, cells: list[Cell], reduce_fn) -> None:
        job.state = "running"
        job.started = time.time()
        results: list = [None] * len(cells)
        try:
            if self._workers <= 1:
                for i, (fn, args) in enumerate(cells):
                    if self._cancelled(job.job_id):
                        job.state = "cancelled"
                        break
                    results[i] = fn(*args)
                    job.progress_done = i + 1
            else:
                pool = self._get_pool()
                pending = {pool.submit(fn, *args): i for i, (fn, args) in enumerate(cells)}
                while pending:
                    done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for fut in done:
                        results[pending.pop(fut)] = fut.result()
                        job.progress_done += 1
                    if self._cancelled(job.job_id):
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
        finally:
            job.finished = time.time()
            self._persist(job)

    # -- persistence ---------------------------------------------------------

    def _persist(self, job: Job) -> None:
        if not self._results_dir:
            return
        os.makedirs(self._results_dir, exist_ok=True)
        path = os.path.join(self._results_dir, f"{job.job_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(job.to_dict(), f, indent=2)

    def history(self) -> list[dict]:
        """Persisted finished jobs (older sessions), excluding in-memory ones."""
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
                    out.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
