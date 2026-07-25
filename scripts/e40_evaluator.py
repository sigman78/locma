"""E40 part A — is gen7 the better `rbeam` EVALUATOR? (search-RoR re-pin candidate)

Pre-registered 2026-07-24, before any run. Follows E39 Gate 0.

E32 Phase 1 closed "swap the reactive champion in as the search evaluator" as
null-to-negative — but it measured **e29slim** (`rbeam` 0.335 vs a `shared`
evaluator), and localized the loss to e29slim's weaker value tower. E39 Gate 0
produced a reason to doubt that generalizes to gen7:

  * E25:  `rbeam:shared` was at PARITY with the EASIER `dmcts:15,100`  (0.501)
  * E39:  `rbeam:gen7` BEATS the HARDER `dmcts:15,150`                 (0.551)

Those are different opponents measured in different rounds, so they are a lead and
not a result. This script settles it head-to-head, which is how `rbeam` itself was
promoted (E24): same wrapper config 8,20,4,4 on both sides, matched `ldraft` both
sides, so the EVALUATOR NET is the only variable.

Two cells, because the incumbent is a 3-critic ensemble and gen7 is one net:

  cheap_1v3   `rbeam:gen7` (1 critic) vs `rbeam:shared` (3 critics)
              The practical question. A win here is also a ~3x evaluator-compute
              win, i.e. strictly better on both axes.
  matched_3v3 `rbeam:<e36 trio>` vs `rbeam:shared` (3 vs 3)
              The fair matched-compute comparison. The e36 trio is the three
              independent PFSP endpoints (x86 14M / m1 20M / s22 22M) — note they
              differ in platform and n_envs as well as seed, so the trio carries
              MORE diversity than the e29slim trio did (E7/E8 diversity thesis
              says that helps an ensemble; flagged, not assumed).

Promotion discipline (this project's E7 precedent): a primary read AND a
fresh-seed confirm, both CI-positive, before anything re-pins. Stage `confirm`
re-runs the winning cell on a disjoint seed base.

Pre-registered reads (candidate WR over `rbeam:shared`):
  CI lo > 0.50 on primary AND confirm  -> re-pin the play-time SEARCH recipe of
                                          record to the gen7 evaluator
  CI straddles 0.50                    -> no re-pin; `shared` stays (E32 stands,
                                          and the E39 lead was opponent artifact)
  CI hi < 0.50                          -> E32's negative DOES generalize to gen7

    python scripts/e40_evaluator.py --stage primary --pairs 200 --workers 8
    python scripts/e40_evaluator.py --stage confirm --cells cheap_1v3 --pairs 200
    python scripts/e40_evaluator.py --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from locma.stats.intervals import wilson_ci

LDRAFT = "depot:ldraft/ldraft_s0.zip"

SHARED_TRIO = "depot:shared/shared_s0.zip|depot:shared/shared_s1.zip|depot:shared/shared_s2.zip"
# The three independent PFSP parity endpoints (seeds 14M x86 / 20M m1 / 22M s22).
E36_TRIO = "depot:e36/e36_gen7.zip|depot:e36m1/e36_m1_gen7.zip|depot:e36s22/e36_s22_gen7.zip"
GEN7 = "depot:e36/e36_gen7.zip"

RB = "8,20,4,4"  # the search recipe of record's wrapper config (E24)
INCUMBENT = f"rbeam:{SHARED_TRIO},{RB},{LDRAFT}"

CELLS = {
    "cheap_1v3": f"rbeam:{GEN7},{RB},{LDRAFT}",
    "matched_3v3": f"rbeam:{E36_TRIO},{RB},{LDRAFT}",
}

SEEDS = {"primary": 82_000_000, "confirm": 83_000_000}

_CACHE: dict = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _cell(a_spec: str, b_spec: str, seed: int, pairs: int) -> tuple[int, int, float]:
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    t0 = time.perf_counter()
    for spec in (a_spec, b_spec):
        if spec not in _CACHE:
            _CACHE[spec] = make_policy(spec)
    res = run_match(_CACHE[a_spec], _CACHE[b_spec], games=pairs, seed=seed)
    return res.wins_a, res.games, time.perf_counter() - t0


def _noop() -> None:
    try:
        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        init_eval_worker()
    except Exception:  # noqa: BLE001
        pass


def verdict(ci: list[float]) -> str:
    if ci[0] > 0.50:
        return "CANDIDATE AHEAD (CI excludes 0.50) — re-pin if the confirm agrees"
    if ci[1] < 0.50:
        return "candidate BEHIND — E32's evaluator negative generalizes to gen7"
    return "PARITY — no re-pin; shared stays the search evaluator of record"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="primary", choices=("primary", "confirm"))
    ap.add_argument("--cells", default=",".join(CELLS), help=f"subset of {list(CELLS)}")
    ap.add_argument("--pairs", type=int, default=200, help="seed pairs/cell (n = 2*pairs)")
    ap.add_argument("--block-pairs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="3 pairs, serial")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    labels = [s.strip() for s in args.cells.split(",") if s.strip()]
    for label in labels:
        if label not in CELLS:
            ap.error(f"unknown cell {label!r}; known: {list(CELLS)}")
    pairs = 3 if args.smoke else args.pairs
    workers = 1 if args.smoke else args.workers
    seed0 = SEEDS[args.stage]
    out = args.out or f"runs/e40/evaluator_{args.stage}.json"

    units = []
    for i, label in enumerate(labels):
        off = 0
        while off < pairs:
            n = min(args.block_pairs, pairs - off)
            units.append((label, CELLS[label], INCUMBENT, seed0 + i * 1000 + off, n))
            off += n

    print(f"E40-A evaluator re-pin — {utc_now()}  stage={args.stage} cells={labels}")
    print(f"  incumbent: {INCUMBENT}")
    print(f"  {len(units)} blocks on {workers} workers", flush=True)

    agg: dict[str, list[float]] = {label: [0, 0, 0.0] for label in labels}
    t0 = time.perf_counter()
    if workers > 1 and len(units) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_noop) as ex:
            futs = {ex.submit(_cell, u[1], u[2], u[3], u[4]): (u[0], u[3]) for u in units}
            for f in as_completed(futs):
                label, seed = futs[f]
                w, g, sec = f.result()
                agg[label][0] += w
                agg[label][1] += g
                agg[label][2] += sec
                print(f"    [{label:12s}] seed {seed}: {w}/{g} ({sec:.0f}s)", flush=True)
    else:
        for u in units:
            w, g, sec = _cell(u[1], u[2], u[3], u[4])
            agg[u[0]][0] += w
            agg[u[0]][1] += g
            agg[u[0]][2] += sec

    cells = {}
    for label in labels:
        w, g, sec = agg[label]
        lo, hi = wilson_ci(int(w), int(g))
        cells[label] = {
            "candidate": CELLS[label],
            "candidate_wr": round(w / g, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "n": int(g),
            "s_per_game": round(sec / g, 2) if g else None,
            "verdict": verdict([lo, hi]),
        }

    payload = {
        "generated": utc_now(),
        "stage": args.stage,
        "seed_base": seed0,
        "incumbent": INCUMBENT,
        "question": "is a gen7 evaluator better than the `shared` trio inside rbeam 8,20,4,4?",
        "cells": cells,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))

    print(f"\n======== E40-A EVALUATOR RE-PIN — {args.stage.upper()} ========")
    for label, v in cells.items():
        print(
            f"{label:12s} {v['candidate_wr']:.4f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] "
            f"(n={v['n']}, {v['s_per_game']}s/game)"
        )
        print(f"{'':12s} -> {v['verdict']}")
    print(f"\nwrote {out}  ({payload['seconds']}s)")


if __name__ == "__main__":
    main()
