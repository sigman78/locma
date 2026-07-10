"""E23: fixed-budget depth allocation for current fair net-guided dMCTS.

Question
--------
E22 showed that real multi-turn depth, rather than hidden-information access,
is what lets MCTS beat the current one-turn vbeam planner.  ``netdmcts`` has
the right ingredients (fair determinizations plus learned policy/value
guidance), but its historical ``K=8, I=40`` recipe fragments 320 neural leaf
evaluations across eight independent, shallow trees.  Does concentrating the
same budget into fewer, deeper trees make it stronger?

Primary fixed-budget sweep (320 evaluations per decision)::

    K x I = 1x320, 2x160, 4x80, 8x40, 16x20

The current planner recipe of record and every candidate use the same
``ldraft_s0`` draft model, isolating battle search.  Two current token oracles
are tested: ``b0k_s0`` (reactive recipe) and ``shared_s0`` (the family whose
critics power the planner recipe).

Stages
------
``pilot`` (default)
    25 seed pairs/cell = 50 actual games after mirroring.  All cells share the
    same seed range (common random numbers).  Results are Wilson 95% intervals
    for netdmcts's direct win rate against the planner.
``confirm``
    Select the best pilot cell (or explicit ``--confirm-cell`` entries) and
    run 100 fresh seed pairs = 200 actual games/cell.
``all``
    Run pilot then confirm in one invocation.

Operational safety
------------------
The run is idempotent and crash-resumable.  Every small seed block is written
under ``runs/e23-blocks/`` as soon as it completes; rerunning skips valid
blocks and reconstructs the summary.  ``--dry-run`` prints the exact cells
without touching artifacts.  ``--preflight-only`` resolves every depot blob,
checks the ML stack/device, and constructs the policy specs without playing.

Recommended commands (PowerShell, from the repository root)::

    # Inspect and validate first.
    uv run --extra ml python scripts/e23_netdmcts_alloc.py --dry-run
    uv run --extra ml python scripts/e23_netdmcts_alloc.py --preflight-only

    # End-to-end tiny plumbing check (2 actual games/cell).
    uv run --extra ml python scripts/e23_netdmcts_alloc.py --smoke --stage all

    # Real pilot. Start with one worker; benchmark 2 before using more because
    # each Windows worker owns four GPU models (candidate + 3-critic planner).
    uv run --extra ml python scripts/e23_netdmcts_alloc.py --stage pilot --workers 1

    # Fresh-seed confirm of the pilot winner.
    uv run --extra ml python scripts/e23_netdmcts_alloc.py --stage confirm --workers 1

At the measured ~33 sec/actual-game for a 320-evaluation netdmcts policy, the
full 10-cell pilot is roughly 4.7 serial GPU-hours before overhead.  Worker
parallelism may or may not help a single GPU; compare the recorded
``worker_seconds_per_game`` rather than assuming CPU-style scaling.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from locma.harness.parallel import init_eval_worker

EXPERIMENT = "E23"
DESIGN_VERSION = 1

ORACLE_REFS = {
    "b0k": "depot:b0k/b0k_s0.zip",
    "shared": "depot:shared/shared_s0.zip",
}
SHARED_REFS = tuple(f"depot:shared/shared_s{s}.zip" for s in (0, 1, 2))
LDRAFT0 = "depot:ldraft/ldraft_s0.zip"
C_PUCT = 1.5
PILOT_SEED0 = 27_000_000
CONFIRM_SEED0 = 28_000_000
DEFAULT_ALLOCATIONS = "1x320,2x160,4x80,8x40,16x20"


@dataclass(frozen=True)
class Cell:
    oracle: str
    k: int
    i: int

    @property
    def key(self) -> str:
        return f"{self.oracle}_k{self.k}_i{self.i}"

    @property
    def total_evals(self) -> int:
        return self.k * self.i


def parse_allocations(raw: str, *, require_fixed_budget: bool = True) -> list[tuple[int, int]]:
    """Parse ``1x320,2x160`` and enforce a positive fixed total budget."""
    out: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        try:
            k_raw, i_raw = token.split("x", 1)
            k, i = int(k_raw), int(i_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bad allocation '{token}' (want KxI, e.g. 2x160)") from exc
        if k < 1 or i < 1:
            raise ValueError(f"allocation '{token}' must have K >= 1 and I >= 1")
        if (k, i) in out:
            raise ValueError(f"duplicate allocation '{token}'")
        out.append((k, i))
    if not out:
        raise ValueError("at least one KxI allocation is required")
    totals = {k * i for k, i in out}
    if require_fixed_budget and len(totals) != 1:
        raise ValueError(f"allocations must have one fixed K*I budget; got {sorted(totals)}")
    return out


def parse_confirm_cell(raw: str) -> Cell:
    """Parse an explicit confirmation selector such as ``b0k:2x160``."""
    try:
        oracle, allocation = raw.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"bad confirm cell '{raw}' (want ORACLE:KxI)") from exc
    if oracle not in ORACLE_REFS:
        raise ValueError(f"unknown oracle '{oracle}' (choose from {sorted(ORACLE_REFS)})")
    allocations = parse_allocations(allocation, require_fixed_budget=False)
    if len(allocations) != 1:
        raise ValueError(f"confirm cell '{raw}' must name exactly one KxI allocation")
    (k, i) = allocations[0]
    return Cell(oracle, k, i)


def planner_spec() -> str:
    return "vbeam:" + "|".join(SHARED_REFS) + f",8,20,{LDRAFT0}"


def candidate_spec(cell: Cell, c_puct: float = C_PUCT) -> str:
    return f"netdmcts:{cell.k},{cell.i},{c_puct},{ORACLE_REFS[cell.oracle]},{LDRAFT0}"


def build_blocks(seed0: int, pairs: int, block_pairs: int) -> list[tuple[int, int, int]]:
    """Return ``(block_index, first_seed, seed_pairs)`` covering a stage exactly."""
    if pairs < 1 or block_pairs < 1:
        raise ValueError("pairs and block_pairs must be >= 1")
    blocks = []
    offset = 0
    while offset < pairs:
        n = min(block_pairs, pairs - offset)
        blocks.append((len(blocks), seed0 + offset, n))
        offset += n
    return blocks


def select_confirm_cells(pilot_cells: dict, top: int = 1) -> list[Cell]:
    """Select pilot leaders by candidate win rate, then lower CI and game count."""
    if top < 1:
        raise ValueError("confirm top count must be >= 1")
    if not pilot_cells:
        raise ValueError("no completed pilot cells found; run --stage pilot first")
    ranked = sorted(
        pilot_cells.values(),
        key=lambda row: (row["candidate_wr"], row["ci_lo"], row["games"]),
        reverse=True,
    )
    return [Cell(row["oracle"], int(row["K"]), int(row["I"])) for row in ranked[:top]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


# One candidate and one planner per worker. When a cell changes, replace only
# that slot so long runs do not retain every historical CUDA model.
_WORKER_POLICIES: dict[str, tuple[str, object]] = {}


def _cached_policy(slot: str, spec: str):
    entry = _WORKER_POLICIES.get(slot)
    if entry is not None and entry[0] == spec:
        return entry[1]
    if entry is not None:
        del _WORKER_POLICIES[slot]
        gc.collect()
    from locma.policies.registry import make_policy  # noqa: PLC0415

    policy = make_policy(spec)
    _WORKER_POLICIES[slot] = (spec, policy)
    return policy


def _match_block(candidate: str, planner: str, seed: int, pairs: int) -> dict:
    """Picklable worker unit; ``pairs`` produces exactly ``2*pairs`` games."""
    from locma.harness.match import run_match  # noqa: PLC0415

    t0 = time.perf_counter()
    cand = _cached_policy("candidate", candidate)
    base = _cached_policy("planner", planner)
    result = run_match(cand, base, games=pairs, seed=seed)
    seat0 = [r for r in result.records if r["a_seat"] == 0]
    seat1 = [r for r in result.records if r["a_seat"] == 1]
    return {
        "candidate": candidate,
        "planner": planner,
        "seed": seed,
        "seed_pairs": pairs,
        "games": result.games,
        "candidate_wins": result.wins_a,
        "planner_wins": result.wins_b,
        "candidate_seat0_wins": sum(bool(r["winner_is_a"]) for r in seat0),
        "candidate_seat1_wins": sum(bool(r["winner_is_a"]) for r in seat1),
        "turns": sum(int(r["turns"]) for r in result.records),
        "worker_seconds": round(time.perf_counter() - t0, 3),
    }


class Driver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.prefix = args.output_prefix
        self.summary_path = Path("runs") / f"{self.prefix}-summary.json"
        self.log_path = Path("runs") / f"{self.prefix}.log"
        self.blocks_root = Path("runs") / f"{self.prefix}-blocks"
        self.summary = self._load_summary()

    def _load_summary(self) -> dict:
        if self.summary_path.is_file():
            with open(self.summary_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("experiment") != EXPERIMENT:
                raise ValueError(f"{self.summary_path} is not an {EXPERIMENT} summary")
            if data.get("design_version") != DESIGN_VERSION:
                raise ValueError(
                    f"{self.summary_path} has design version {data.get('design_version')}, "
                    f"expected {DESIGN_VERSION}"
                )
            expected = {
                "planner": planner_spec(),
                "draft": LDRAFT0,
                "c_puct": self.args.c_puct,
            }
            mismatches = {k: (data.get(k), v) for k, v in expected.items() if data.get(k) != v}
            if mismatches:
                raise ValueError(f"{self.summary_path} is incompatible with this run: {mismatches}")
            return data
        return {
            "experiment": EXPERIMENT,
            "design_version": DESIGN_VERSION,
            "created": utc_now(),
            "hypothesis": "At fixed K*I, fewer determinizations and deeper trees improve play.",
            "planner": planner_spec(),
            "draft": LDRAFT0,
            "c_puct": self.args.c_puct,
            "pilot": {"cells": {}},
            "confirm": {"cells": {}},
            "runtime_history": [],
        }

    def log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    def save(self) -> None:
        atomic_json(self.summary_path, self.summary)

    def block_path(self, stage: str, cell: Cell, block_index: int) -> Path:
        return self.blocks_root / stage / f"{cell.key}_block{block_index:03d}.json"

    def load_valid_block(
        self,
        path: Path,
        *,
        candidate: str,
        planner: str,
        seed: int,
        pairs: int,
    ) -> dict | None:
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        expected = {
            "candidate": candidate,
            "planner": planner,
            "seed": seed,
            "seed_pairs": pairs,
            "games": pairs * 2,
        }
        mismatches = {k: (row.get(k), v) for k, v in expected.items() if row.get(k) != v}
        if mismatches:
            raise ValueError(f"stale/incompatible block {path}: {mismatches}")
        return row

    def _run_missing_blocks(
        self,
        missing: list[tuple[int, int, int, Path]],
        candidate: str,
        planner: str,
        executor: ProcessPoolExecutor | None,
    ) -> None:
        if not missing:
            return
        if executor is None:
            for block_index, seed, pairs, path in missing:
                self.log(f"  block {block_index}: seeds {seed}..{seed + pairs - 1}")
                row = _match_block(candidate, planner, seed, pairs)
                row["completed"] = utc_now()
                atomic_json(path, row)
            return

        future_to_block = {
            executor.submit(_match_block, candidate, planner, seed, pairs): (
                block_index,
                seed,
                pairs,
                path,
            )
            for block_index, seed, pairs, path in missing
        }
        for future in as_completed(future_to_block):
            block_index, seed, pairs, path = future_to_block[future]
            row = future.result()
            row["completed"] = utc_now()
            atomic_json(path, row)
            self.log(
                f"  block {block_index} complete: "
                f"{row['candidate_wins']}/{row['games']} candidate wins"
            )

    def run_cell(
        self,
        stage: str,
        cell: Cell,
        *,
        seed0: int,
        pairs: int,
        block_pairs: int,
        executor: ProcessPoolExecutor | None,
    ) -> dict:
        from locma.stats.intervals import binomial_test, wilson_ci  # noqa: PLC0415

        candidate = candidate_spec(cell, self.args.c_puct)
        planner = planner_spec()
        blocks = build_blocks(seed0, pairs, block_pairs)
        completed: list[dict] = []
        missing: list[tuple[int, int, int, Path]] = []
        for block_index, seed, count in blocks:
            path = self.block_path(stage, cell, block_index)
            row = self.load_valid_block(
                path,
                candidate=candidate,
                planner=planner,
                seed=seed,
                pairs=count,
            )
            if row is None:
                missing.append((block_index, seed, count, path))
            else:
                completed.append(row)

        self.log(
            f"{stage} {cell.key}: K*I={cell.total_evals}, {pairs} seed pairs / "
            f"{2 * pairs} actual games ({len(completed)}/{len(blocks)} blocks cached)"
        )
        wall0 = time.perf_counter()
        self._run_missing_blocks(missing, candidate, planner, executor)
        invocation_wall = time.perf_counter() - wall0

        # Reload in canonical block order, including blocks just checkpointed.
        rows = []
        for block_index, seed, count in blocks:
            path = self.block_path(stage, cell, block_index)
            row = self.load_valid_block(
                path,
                candidate=candidate,
                planner=planner,
                seed=seed,
                pairs=count,
            )
            assert row is not None
            rows.append(row)

        games = sum(int(r["games"]) for r in rows)
        wins = sum(int(r["candidate_wins"]) for r in rows)
        lo, hi = wilson_ci(wins, games)
        worker_seconds = sum(float(r["worker_seconds"]) for r in rows)
        out = {
            "oracle": cell.oracle,
            "oracle_ref": ORACLE_REFS[cell.oracle],
            "K": cell.k,
            "I": cell.i,
            "total_evals": cell.total_evals,
            "candidate": candidate,
            "planner": planner,
            "seed0": seed0,
            "seed_pairs": pairs,
            "games": games,
            "candidate_wins": wins,
            "planner_wins": games - wins,
            "candidate_wr": round(wins / games, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "p_two_sided": round(binomial_test(wins, games), 6),
            "candidate_ahead": bool(lo > 0.5),
            "planner_ahead": bool(hi < 0.5),
            "candidate_seat0_wr": round(
                sum(int(r["candidate_seat0_wins"]) for r in rows) / pairs, 4
            ),
            "candidate_seat1_wr": round(
                sum(int(r["candidate_seat1_wins"]) for r in rows) / pairs, 4
            ),
            "mean_turns": round(sum(int(r["turns"]) for r in rows) / games, 2),
            "worker_seconds_per_game": round(worker_seconds / games, 3),
            "new_wall_minutes": round(invocation_wall / 60, 2),
            "completed": utc_now(),
        }
        self.summary[stage]["cells"][cell.key] = out
        self.save()
        self.log(
            f"{stage} {cell.key} DONE: wr={out['candidate_wr']:.4f} "
            f"CI=[{out['ci_lo']:.4f},{out['ci_hi']:.4f}], "
            f"{out['worker_seconds_per_game']:.1f} worker-sec/game"
        )
        return out

    def run_stage(
        self,
        stage: str,
        cells: list[Cell],
        *,
        seed0: int,
        pairs: int,
        block_pairs: int,
    ) -> list[dict]:
        executor = None
        if self.args.workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=self.args.workers,
                initializer=init_eval_worker,
            )
        try:
            return [
                self.run_cell(
                    stage,
                    cell,
                    seed0=seed0,
                    pairs=pairs,
                    block_pairs=block_pairs,
                    executor=executor,
                )
                for cell in cells
            ]
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    def finish_read(self) -> None:
        pilot = self.summary["pilot"]["cells"]
        confirm = self.summary["confirm"]["cells"]
        read: dict = {"generated": utc_now()}
        if pilot:
            winner = select_confirm_cells(pilot, 1)[0]
            read["pilot_winner"] = pilot[winner.key]
            read["pilot_ranking"] = [
                row["oracle"] + f":{row['K']}x{row['I']}"
                for row in sorted(
                    pilot.values(), key=lambda x: (x["candidate_wr"], x["ci_lo"]), reverse=True
                )
            ]
        if confirm:
            read["confirm"] = list(confirm.values())
            read["confirmed_headroom"] = any(row["candidate_ahead"] for row in confirm.values())
        self.summary["read"] = read
        self.save()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E23 fixed-budget K/I allocation sweep for current netdmcts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=("pilot", "confirm", "all"), default="pilot")
    parser.add_argument(
        "--oracles",
        nargs="+",
        choices=tuple(ORACLE_REFS),
        default=list(ORACLE_REFS),
        help="oracle families included in the pilot",
    )
    parser.add_argument("--allocations", default=DEFAULT_ALLOCATIONS, help="comma-separated KxI")
    parser.add_argument("--c-puct", type=float, default=C_PUCT)
    parser.add_argument("--workers", type=int, default=1, help="GPU-owning evaluation workers")
    parser.add_argument("--pilot-pairs", type=int, default=25, help="mirrored seed pairs per cell")
    parser.add_argument("--confirm-pairs", type=int, default=100, help="fresh seed pairs per cell")
    parser.add_argument("--block-pairs", type=int, default=5, help="checkpoint granularity")
    parser.add_argument("--pilot-seed", type=int, default=PILOT_SEED0)
    parser.add_argument("--confirm-seed", type=int, default=CONFIRM_SEED0)
    parser.add_argument("--confirm-top", type=int, default=1, help="pilot leaders to confirm")
    parser.add_argument(
        "--confirm-cell",
        action="append",
        default=[],
        metavar="ORACLE:KxI",
        help="explicit confirmation cell; repeatable and overrides --confirm-top",
    )
    parser.add_argument("--output-prefix", default="e23", help="runs/ artifact prefix")
    parser.add_argument("--smoke", action="store_true", help="two tiny 2-evaluation cells")
    parser.add_argument(
        "--dry-run", action="store_true", help="print plan without resolving models"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="resolve artifacts and check ML/device without games",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1 (GPU auto-scaling is intentionally disabled)")
    for name in ("pilot_pairs", "confirm_pairs", "block_pairs", "confirm_top"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.c_puct <= 0:
        raise ValueError("--c-puct must be > 0")
    if args.smoke:
        args.allocations = "1x2,2x1"
        args.oracles = ["b0k"]
        args.pilot_pairs = 1
        args.confirm_pairs = 1
        args.block_pairs = 1
        if args.output_prefix == "e23":
            args.output_prefix = "e23-smoke"
    parse_allocations(args.allocations)
    for raw in args.confirm_cell:
        parse_confirm_cell(raw)


def planned_pilot_cells(args: argparse.Namespace) -> list[Cell]:
    allocations = parse_allocations(args.allocations)
    return [Cell(oracle, k, i) for oracle in args.oracles for k, i in allocations]


def print_plan(args: argparse.Namespace) -> None:
    pilot = planned_pilot_cells(args)
    print(f"{EXPERIMENT} planner: {planner_spec()}")
    print(f"pilot: {len(pilot)} cells, {2 * args.pilot_pairs} actual games/cell")
    for cell in pilot:
        print(f"  {cell.key}: {candidate_spec(cell, args.c_puct)}")
    actual_games = len(pilot) * 2 * args.pilot_pairs
    serial_hours = actual_games * 33 / 3600
    print(f"pilot total: {actual_games} actual games; rough serial estimate {serial_hours:.1f} h")
    if args.confirm_cell:
        confirm = [parse_confirm_cell(raw) for raw in args.confirm_cell]
        print("explicit confirm: " + ", ".join(cell.key for cell in confirm))
    else:
        print(f"confirm: top {args.confirm_top} pilot cell(s), {2 * args.confirm_pairs} games/cell")
    print(f"outputs: runs/{args.output_prefix}-summary.json, .log, and -blocks/")


def preflight(args: argparse.Namespace) -> dict:
    """Resolve required artifacts and validate policy construction without inference."""
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    try:
        import sb3_contrib  # noqa: F401, PLC0415
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("E23 requires the [ml] extra: uv sync --extra ml") from exc

    # Resolve both oracle families even for a confirm-only invocation: its
    # selected pilot winner may not be in a narrowed --oracles argument.
    refs = [LDRAFT0, *SHARED_REFS, *ORACLE_REFS.values()]
    resolved = {ref: resolve_path(ref) for ref in dict.fromkeys(refs)}
    for path in resolved.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    # Construction exercises registry parsing, including E23's fifth draft
    # parameter. Models remain lazy, so this does not allocate four networks.
    make_policy(planner_spec())
    for cell in planned_pilot_cells(args):
        make_policy(candidate_spec(cell, args.c_puct))
    for raw in args.confirm_cell:
        make_policy(candidate_spec(parse_confirm_cell(raw), args.c_puct))

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.get_device_name(0)}"
    out = {"device": device, "artifacts": resolved, "cells": len(planned_pilot_cells(args))}
    print(json.dumps(out, indent=2))
    if device == "cpu":
        print("WARNING: CUDA is unavailable; the real E23 grid will be extremely slow.")
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    print_plan(args)
    if args.dry_run:
        return 0
    preflight_info = preflight(args)
    if args.preflight_only:
        return 0

    Path("runs").mkdir(exist_ok=True)
    driver = Driver(args)
    driver.summary["runtime_history"].append(
        {
            "started": utc_now(),
            "argv": sys.argv,
            "stage": args.stage,
            "workers": args.workers,
            "preflight": preflight_info,
        }
    )
    driver.save()
    driver.log(f"=== {EXPERIMENT} netdmcts allocation {args.stage} start ===")

    if args.stage in ("pilot", "all"):
        driver.run_stage(
            "pilot",
            planned_pilot_cells(args),
            seed0=args.pilot_seed,
            pairs=args.pilot_pairs,
            block_pairs=args.block_pairs,
        )

    if args.stage in ("confirm", "all"):
        if args.confirm_cell:
            confirm_cells = [parse_confirm_cell(raw) for raw in args.confirm_cell]
        else:
            confirm_cells = select_confirm_cells(driver.summary["pilot"]["cells"], args.confirm_top)
        driver.run_stage(
            "confirm",
            confirm_cells,
            seed0=args.confirm_seed,
            pairs=args.confirm_pairs,
            block_pairs=args.block_pairs,
        )

    driver.finish_read()
    driver.log(f"=== {EXPERIMENT} netdmcts allocation {args.stage} DONE ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("DRIVER CRASHED:\n" + traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1) from exc
