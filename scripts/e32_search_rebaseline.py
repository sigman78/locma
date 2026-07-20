"""E32 Phase 1: search re-baseline on depot:e29slim — the evaluator swap.

Question
--------
The play-time-search recipes of record (rbeam, netdmcts) use the `shared` critic
ensemble as their evaluator net. `depot:e29slim` is a stronger REACTIVE net
(0.903 pure) but was bred for the policy/pointer head, and the E29 slim re-probe
found its VALUE tower marginally weaker on winner_side (0.922 vs 0.932). Search
leans on the value head. So swapping the evaluator shared -> e29slim is genuinely
uncertain: does the stronger reactive base make a better or worse search
evaluator, at a MATCHED search config?

Pre-registered prediction: netdmcts (policy-prior/PUCT-driven -> uses e29slim's
STRONGER pointer head) benefits >= rbeam (value-driven -> the marginally weaker
value tower). Phase 0 confirmed all wrappers run with the e29slim evaluator and
that it is ~cost-parity at the deep RoR configs (netdmcts 0.84x, rbeam 1.26x).

Design
------
Head-to-head, matched config, matched ldraft draft BOTH sides (the only
difference is the evaluator net). run_match mirrors seats, so a self-swap would
sit at 0.500. candidate = e29slim evaluator, baseline = shared evaluator.

  rbeam:    rbeam:<trio>,8,20,4,4,<ldraft>       (3-net ensemble, the RoR config)
  netdmcts: netdmcts:1,320,1.5,<net_s0>,<ldraft> (single net, the prior record)

Stages: pilot 100 seed pairs (200 games), confirm 250 pairs (500 games) — the
E24 pattern. Idempotent, crash-resumable per seed block under runs/e32-blocks/.

Commands (from repo root)::

    .venv/Scripts/python scripts/e32_search_rebaseline.py --dry-run
    .venv/Scripts/python scripts/e32_search_rebaseline.py --smoke
    .venv/Scripts/python scripts/e32_search_rebaseline.py --arm both --stage pilot --workers 4
    .venv/Scripts/python scripts/e32_search_rebaseline.py --arm netdmcts --stage confirm --workers 4
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from locma.harness.parallel import init_eval_worker
from locma.stats.intervals import wilson_ci

E29 = "depot:e29slim/e29slim_s0.zip|depot:e29slim/e29slim_s1.zip|depot:e29slim/e29slim_s2.zip"
SHARED = "depot:shared/shared_s0.zip|depot:shared/shared_s1.zip|depot:shared/shared_s2.zip"
LDRAFT = "depot:ldraft/ldraft_s0.zip"
SEED0 = 30_000_000  # the E22/E23/E24 head-to-head ruler seed base

# arm -> (candidate_spec [e29slim eval], baseline_spec [shared eval])
ARMS = {
    "rbeam": (
        f"rbeam:{E29},8,20,4,4,{LDRAFT}",
        f"rbeam:{SHARED},8,20,4,4,{LDRAFT}",
    ),
    "netdmcts": (
        f"netdmcts:1,320,1.5,depot:e29slim/e29slim_s0.zip,{LDRAFT}",
        f"netdmcts:1,320,1.5,depot:shared/shared_s0.zip,{LDRAFT}",
    ),
}
STAGE_PAIRS = {"pilot": 100, "confirm": 250}

_WORKER_POLICIES: dict[str, tuple[str, object]] = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    tmp.replace(path)


def build_blocks(seed0: int, pairs: int, block_pairs: int) -> list[tuple[int, int, int]]:
    """(block_index, first_seed, seed_pairs) tiling exactly ``pairs`` seed pairs."""
    blocks = []
    offset = 0
    while offset < pairs:
        n = min(block_pairs, pairs - offset)
        blocks.append((len(blocks), seed0 + offset, n))
        offset += n
    return blocks


def _cached_policy(slot: str, spec: str):
    entry = _WORKER_POLICIES.get(slot)
    if entry is not None and entry[0] == spec:
        return entry[1]
    if entry is not None:
        del _WORKER_POLICIES[slot]
        gc.collect()
    from locma.policies.registry import make_policy  # noqa: PLC0415

    _WORKER_POLICIES[slot] = (spec, make_policy(spec))
    return _WORKER_POLICIES[slot][1]


def _match_block(candidate: str, baseline: str, seed: int, pairs: int) -> dict:
    """Picklable worker unit; ``pairs`` seed pairs -> ``2*pairs`` mirrored games."""
    from locma.harness.match import run_match  # noqa: PLC0415

    t0 = time.perf_counter()
    cand = _cached_policy("candidate", candidate)
    base = _cached_policy("baseline", baseline)
    result = run_match(cand, base, games=pairs, seed=seed)
    seat0 = [r for r in result.records if r["a_seat"] == 0]
    seat1 = [r for r in result.records if r["a_seat"] == 1]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "seed": seed,
        "seed_pairs": pairs,
        "games": result.games,
        "candidate_wins": result.wins_a,
        "baseline_wins": result.wins_b,
        "candidate_seat0_wins": sum(bool(r["winner_is_a"]) for r in seat0),
        "candidate_seat1_wins": sum(bool(r["winner_is_a"]) for r in seat1),
        "worker_seconds": round(time.perf_counter() - t0, 3),
    }


def _valid_block(path: Path, candidate: str, baseline: str, seed: int, pairs: int) -> dict | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    ok = (
        row.get("candidate") == candidate
        and row.get("baseline") == baseline
        and row.get("seed") == seed
        and row.get("seed_pairs") == pairs
    )
    return row if ok else None


def run_arm(arm: str, stage: str, pairs: int, block_pairs: int, workers: int, log) -> dict:
    candidate, baseline = ARMS[arm]
    blocks_dir = Path("runs") / "e32-blocks" / stage
    blocks = build_blocks(SEED0, pairs, block_pairs)
    log(f"[{arm}/{stage}] candidate={candidate}")
    log(f"[{arm}/{stage}] baseline ={baseline}")
    log(f"[{arm}/{stage}] {pairs} seed pairs ({2 * pairs} games) in {len(blocks)} blocks")

    rows: dict[int, dict] = {}
    missing = []
    for bi, seed, n in blocks:
        path = blocks_dir / f"{arm}_block{bi:03d}.json"
        cached = _valid_block(path, candidate, baseline, seed, n)
        if cached is not None:
            rows[bi] = cached
            log(f"  block {bi}: cached (seeds {seed}..{seed + n - 1})")
        else:
            missing.append((bi, seed, n, path))

    def _store(bi: int, path: Path, row: dict) -> None:
        atomic_json(path, row)
        rows[bi] = row
        cw, g = row["candidate_wins"], row["games"]
        log(f"  block {bi} done: cand {cw}/{g} ({cw / g:.3f}), {row['worker_seconds']}s")

    if missing:
        if workers > 1 and len(missing) > 1:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(missing)), initializer=init_eval_worker
            ) as ex:
                fut = {
                    ex.submit(_match_block, candidate, baseline, seed, n): (bi, path)
                    for bi, seed, n, path in missing
                }
                for f in as_completed(fut):
                    bi, path = fut[f]
                    _store(bi, path, f.result())
        else:
            for bi, seed, n, path in missing:
                log(f"  block {bi}: seeds {seed}..{seed + n - 1}")
                _store(bi, path, _match_block(candidate, baseline, seed, n))

    cw = sum(r["candidate_wins"] for r in rows.values())
    g = sum(r["games"] for r in rows.values())
    secs = sum(r["worker_seconds"] for r in rows.values())
    lo, hi = wilson_ci(cw, g)
    s0 = sum(r["candidate_seat0_wins"] for r in rows.values())
    s1 = sum(r["candidate_seat1_wins"] for r in rows.values())
    summary = {
        "arm": arm,
        "stage": stage,
        "candidate": candidate,
        "baseline": baseline,
        "games": g,
        "candidate_wins": cw,
        "candidate_wr": round(cw / g, 4),
        "wilson_ci": [round(lo, 4), round(hi, 4)],
        "candidate_ahead": lo > 0.5,
        "candidate_behind": hi < 0.5,
        "seat0_wr": round(s0 / (g // 2), 4) if g else None,
        "seat1_wr": round(s1 / (g // 2), 4) if g else None,
        "s_per_game": round(secs / g, 2) if g else None,
    }
    verdict = (
        "AHEAD"
        if summary["candidate_ahead"]
        else ("BEHIND" if summary["candidate_behind"] else "TIE")
    )
    log(
        f"[{arm}/{stage}] e29slim vs shared: {summary['candidate_wr']:.3f} "
        f"CI{summary['wilson_ci']} -> {verdict}  (seat0 {summary['seat0_wr']} / "
        f"seat1 {summary['seat1_wr']}, {summary['s_per_game']}s/game)"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=[*ARMS, "both"], default="both")
    ap.add_argument("--stage", choices=[*STAGE_PAIRS], default="pilot")
    ap.add_argument("--pairs", type=int, default=None, help="override seed pairs for the stage")
    ap.add_argument("--block-pairs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="2 pairs/arm, serial")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs/e32-search-rebaseline.json")
    args = ap.parse_args()

    arms = list(ARMS) if args.arm == "both" else [args.arm]
    pairs = 2 if args.smoke else (args.pairs or STAGE_PAIRS[args.stage])
    workers = 1 if args.smoke else args.workers
    stage = "smoke" if args.smoke else args.stage

    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    if args.dry_run:
        for arm in arms:
            c, b = ARMS[arm]
            print(f"[{arm}] {2 * pairs} games / {stage}\n  cand: {c}\n  base: {b}")
        return

    log(f"E32 search re-baseline (evaluator swap) — {utc_now()}")
    results = {arm: run_arm(arm, stage, pairs, args.block_pairs, workers, log) for arm in arms}

    payload = {
        "generated": utc_now(),
        "stage": stage,
        "seed0": SEED0,
        "arms": results,
        "log": lines,
    }
    atomic_json(Path(args.out), payload)
    print(f"\nwrote {args.out}")
    print("\n================ E32 PHASE 1 SUMMARY ================")
    for arm, s in results.items():
        verdict = (
            "AHEAD" if s["candidate_ahead"] else ("BEHIND" if s["candidate_behind"] else "TIE")
        )
        print(
            f"{arm:9s} e29slim vs shared  {s['candidate_wr']:.3f}  CI{s['wilson_ci']}  "
            f"{verdict:6s}  {s['s_per_game']}s/game  (n={s['games']})"
        )


if __name__ == "__main__":
    main()
