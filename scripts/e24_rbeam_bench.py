"""E24: benchmark the reply-aware turn beam (rbeam) against the vbeam planner.

Question
--------
E22/E23 showed that real multi-turn depth is what beats the one-turn ``vbeam``
planner. ``rbeam`` adds exactly one genuine opponent-reply ply on top of
``vbeam``: turn-level expectiminimax over the beam's top ``n_plans`` own-turn
plans, averaged across ``n_worlds`` fair determinizations. Does that single ply
buy a head-to-head win over the current planner recipe of record, and at what
cost per game?

Both sides use the same 3-critic ``shared`` ensemble and the same ``ldraft``
draft, so the ONLY difference is rbeam's reply ply -- a clean isolation of the
added search, exactly like E23 isolated the netdmcts K/I allocation. Unlike E23,
this is NOT a fixed-budget reallocation: cost grows with ``n_plans * n_worlds``,
so the sweep maps the strength/cost Pareto frontier rather than a fixed budget.

Success gate (docs/notes/plans-2026-07-09.md, Priority 2): candidate win rate
> 0.55 head-to-head vs the vbeam recipe AND worker-seconds/game within the cost
gate (default 10 s/game). ``candidate_ahead`` (Wilson CI entirely above 0.5)
is the stronger, sample-size-aware version of the > 0.55 bar.

Stages mirror E23::

    pilot    25 seed pairs/cell = 50 mirrored games; common random seeds.
    confirm  the pilot leader (within the cost gate) on 100 fresh seed pairs.
    all      pilot then confirm.

Operational safety is identical to E23: idempotent, crash-resumable per seed
block under ``runs/e24-blocks/``; ``--dry-run`` / ``--preflight-only`` /
``--smoke`` behave the same.

Recommended commands (PowerShell, from the repository root)::

    uv run --extra ml python scripts/e24_rbeam_bench.py --dry-run
    uv run --extra ml python scripts/e24_rbeam_bench.py --preflight-only
    uv run --extra ml python scripts/e24_rbeam_bench.py --smoke --stage all
    uv run --extra ml python scripts/e24_rbeam_bench.py --stage pilot --workers 1
    uv run --extra ml python scripts/e24_rbeam_bench.py --stage confirm --workers 1
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

EXPERIMENT = "E24"
DESIGN_VERSION = 1

SHARED_REFS = tuple(f"depot:shared/shared_s{s}.zip" for s in (0, 1, 2))
LDRAFT0 = "depot:ldraft/ldraft_s0.zip"
BEAM_WIDTH = 8
BEAM_MAX_ACTIONS = 20
PILOT_SEED0 = 29_000_000
CONFIRM_SEED0 = 30_000_000
# (n_plans, n_worlds) allocations along and across the diagonal: how does the
# reply ply's strength trade off against its n_plans*n_worlds opponent-beam cost.
DEFAULT_ALLOCATIONS = "2x2,3x3,4x4,2x4,4x2"
DEFAULT_COST_GATE = 10.0  # worker-seconds/game the plan's Priority-2 gate allows


@dataclass(frozen=True)
class Cell:
    n_plans: int
    n_worlds: int

    @property
    def key(self) -> str:
        return f"p{self.n_plans}_w{self.n_worlds}"

    @property
    def label(self) -> str:
        return f"{self.n_plans}x{self.n_worlds}"

    @property
    def reply_beams(self) -> int:
        """Opponent-reply beams per decision -- the cost driver."""
        return self.n_plans * self.n_worlds


def parse_allocations(raw: str) -> list[tuple[int, int]]:
    """Parse ``2x2,3x3`` into ``[(n_plans, n_worlds), ...]`` (no fixed budget)."""
    out: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        try:
            p_raw, w_raw = token.split("x", 1)
            p, w = int(p_raw), int(w_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bad allocation '{token}' (want PxW, e.g. 3x3)") from exc
        if p < 1 or w < 1:
            raise ValueError(f"allocation '{token}' must have n_plans >= 1 and n_worlds >= 1")
        if (p, w) in out:
            raise ValueError(f"duplicate allocation '{token}'")
        out.append((p, w))
    if not out:
        raise ValueError("at least one PxW allocation is required")
    return out


def parse_confirm_cell(raw: str) -> Cell:
    """Parse an explicit confirmation selector such as ``3x3``."""
    allocations = parse_allocations(raw)
    if len(allocations) != 1:
        raise ValueError(f"confirm cell '{raw}' must name exactly one PxW allocation")
    (p, w) = allocations[0]
    return Cell(p, w)


def planner_spec() -> str:
    return "vbeam:" + "|".join(SHARED_REFS) + f",{BEAM_WIDTH},{BEAM_MAX_ACTIONS},{LDRAFT0}"


def candidate_spec(cell: Cell) -> str:
    ens = "|".join(SHARED_REFS)
    return f"rbeam:{ens},{BEAM_WIDTH},{BEAM_MAX_ACTIONS},{cell.n_plans},{cell.n_worlds},{LDRAFT0}"


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
    """Select pilot leaders, preferring cells within the cost gate.

    Sort key: cost-gate pass first (so a fast-enough strong cell beats a faster
    but weaker or a stronger but too-slow one), then win rate, then lower CI,
    then fewer opponent beams (cheaper), then game count. This means the
    confirmed cell is one we could actually deploy under the Priority-2 gate.
    """
    if top < 1:
        raise ValueError("confirm top count must be >= 1")
    if not pilot_cells:
        raise ValueError("no completed pilot cells found; run --stage pilot first")
    ranked = sorted(
        pilot_cells.values(),
        key=lambda row: (
            bool(row.get("within_cost_gate", True)),
            row["candidate_wr"],
            row["ci_lo"],
            -int(row["reply_beams"]),
            row["games"],
        ),
        reverse=True,
    )
    return [Cell(int(row["n_plans"]), int(row["n_worlds"])) for row in ranked[:top]]


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
            expected = {"planner": planner_spec(), "draft": LDRAFT0}
            mismatches = {k: (data.get(k), v) for k, v in expected.items() if data.get(k) != v}
            if mismatches:
                raise ValueError(f"{self.summary_path} is incompatible with this run: {mismatches}")
            return data
        return {
            "experiment": EXPERIMENT,
            "design_version": DESIGN_VERSION,
            "created": utc_now(),
            "hypothesis": "One opponent-reply ply lets rbeam beat the one-turn vbeam planner.",
            "planner": planner_spec(),
            "draft": LDRAFT0,
            "cost_gate_seconds": self.args.cost_gate,
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
        self, path: Path, *, candidate: str, planner: str, seed: int, pairs: int
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

        candidate = candidate_spec(cell)
        planner = planner_spec()
        blocks = build_blocks(seed0, pairs, block_pairs)
        missing: list[tuple[int, int, int, Path]] = []
        cached = 0
        for block_index, seed, count in blocks:
            path = self.block_path(stage, cell, block_index)
            row = self.load_valid_block(
                path, candidate=candidate, planner=planner, seed=seed, pairs=count
            )
            if row is None:
                missing.append((block_index, seed, count, path))
            else:
                cached += 1

        self.log(
            f"{stage} {cell.label}: {cell.reply_beams} reply-beams/turn, {pairs} seed pairs / "
            f"{2 * pairs} actual games ({cached}/{len(blocks)} blocks cached)"
        )
        wall0 = time.perf_counter()
        self._run_missing_blocks(missing, candidate, planner, executor)
        invocation_wall = time.perf_counter() - wall0

        rows = []
        for block_index, seed, count in blocks:
            path = self.block_path(stage, cell, block_index)
            row = self.load_valid_block(
                path, candidate=candidate, planner=planner, seed=seed, pairs=count
            )
            assert row is not None
            rows.append(row)

        games = sum(int(r["games"]) for r in rows)
        wins = sum(int(r["candidate_wins"]) for r in rows)
        lo, hi = wilson_ci(wins, games)
        worker_seconds = sum(float(r["worker_seconds"]) for r in rows)
        sec_per_game = worker_seconds / games
        out = {
            "n_plans": cell.n_plans,
            "n_worlds": cell.n_worlds,
            "reply_beams": cell.reply_beams,
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
            "beats_55": bool(wins / games > 0.55),
            "within_cost_gate": bool(sec_per_game <= self.args.cost_gate),
            "candidate_seat0_wr": round(
                sum(int(r["candidate_seat0_wins"]) for r in rows) / pairs, 4
            ),
            "candidate_seat1_wr": round(
                sum(int(r["candidate_seat1_wins"]) for r in rows) / pairs, 4
            ),
            "mean_turns": round(sum(int(r["turns"]) for r in rows) / games, 2),
            "worker_seconds_per_game": round(sec_per_game, 3),
            "new_wall_minutes": round(invocation_wall / 60, 2),
            "completed": utc_now(),
        }
        self.summary[stage]["cells"][cell.key] = out
        self.save()
        self.log(
            f"{stage} {cell.label} DONE: wr={out['candidate_wr']:.4f} "
            f"CI=[{out['ci_lo']:.4f},{out['ci_hi']:.4f}], "
            f"{out['worker_seconds_per_game']:.1f} s/game "
            f"(gate {'PASS' if out['within_cost_gate'] else 'FAIL'})"
        )
        return out

    def run_stage(
        self, stage: str, cells: list[Cell], *, seed0: int, pairs: int, block_pairs: int
    ) -> list[dict]:
        executor = None
        if self.args.workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=self.args.workers, initializer=init_eval_worker
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
                f"{row['n_plans']}x{row['n_worlds']}"
                for row in sorted(
                    pilot.values(), key=lambda x: (x["candidate_wr"], x["ci_lo"]), reverse=True
                )
            ]
        if confirm:
            read["confirm"] = list(confirm.values())
            # The plan's gate: CI entirely above 0.5 AND within the cost gate.
            read["confirmed_headroom"] = any(
                row["candidate_ahead"] and row["within_cost_gate"] for row in confirm.values()
            )
        self.summary["read"] = read
        self.save()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E24 rbeam (reply-aware turn beam) vs the vbeam planner recipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=("pilot", "confirm", "all"), default="pilot")
    parser.add_argument(
        "--allocations",
        default=DEFAULT_ALLOCATIONS,
        help="comma-separated PxW (n_plans x n_worlds)",
    )
    parser.add_argument(
        "--cost-gate", type=float, default=DEFAULT_COST_GATE, help="max worker-seconds/game to pass"
    )
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
        metavar="PxW",
        help="explicit confirmation cell; repeatable and overrides --confirm-top",
    )
    parser.add_argument("--output-prefix", default="e24", help="runs/ artifact prefix")
    parser.add_argument("--smoke", action="store_true", help="two tiny cells, 1 pair each")
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
    if args.cost_gate <= 0:
        raise ValueError("--cost-gate must be > 0")
    if args.smoke:
        args.allocations = "1x1,2x1"
        args.pilot_pairs = 1
        args.confirm_pairs = 1
        args.block_pairs = 1
        if args.output_prefix == "e24":
            args.output_prefix = "e24-smoke"
    parse_allocations(args.allocations)
    for raw in args.confirm_cell:
        parse_confirm_cell(raw)


def planned_pilot_cells(args: argparse.Namespace) -> list[Cell]:
    return [Cell(p, w) for p, w in parse_allocations(args.allocations)]


def print_plan(args: argparse.Namespace) -> None:
    pilot = planned_pilot_cells(args)
    print(f"{EXPERIMENT} planner: {planner_spec()}")
    print(f"pilot: {len(pilot)} cells, {2 * args.pilot_pairs} actual games/cell")
    for cell in pilot:
        print(f"  {cell.key}: {candidate_spec(cell)}")
    actual_games = len(pilot) * 2 * args.pilot_pairs
    # Rough serial estimate: ~1 s base + ~0.35 s per opponent-beam per game.
    est_sec = sum(2 * args.pilot_pairs * (1.0 + 0.35 * c.reply_beams) for c in pilot)
    print(
        f"pilot total: {actual_games} actual games; rough serial estimate "
        f"{est_sec / 3600:.1f} h; cost gate {args.cost_gate:.0f} s/game"
    )
    if args.confirm_cell:
        confirm = [parse_confirm_cell(raw) for raw in args.confirm_cell]
        print("explicit confirm: " + ", ".join(cell.label for cell in confirm))
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
        raise RuntimeError("E24 requires the [ml] extra: uv sync --extra ml") from exc

    refs = [LDRAFT0, *SHARED_REFS]
    resolved = {ref: resolve_path(ref) for ref in dict.fromkeys(refs)}
    for path in resolved.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    make_policy(planner_spec())
    for cell in planned_pilot_cells(args):
        make_policy(candidate_spec(cell))
    for raw in args.confirm_cell:
        make_policy(candidate_spec(parse_confirm_cell(raw)))

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.get_device_name(0)}"
    out = {"device": device, "artifacts": resolved, "cells": len(planned_pilot_cells(args))}
    print(json.dumps(out, indent=2))
    if device == "cpu":
        print("WARNING: CUDA is unavailable; the real E24 grid will be extremely slow.")
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
    driver.log(f"=== {EXPERIMENT} rbeam benchmark {args.stage} start ===")

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
    driver.log(f"=== {EXPERIMENT} rbeam benchmark {args.stage} DONE ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("DRIVER CRASHED:\n" + traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1) from exc
