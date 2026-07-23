"""E36 draft affinity: how much does the battle net's win rate depend on the deck?

Holds the OPPONENT fixed — a stable, ~equally-powerful fair search oracle
(``dmcts:15,60`` with a constant ``depot:ldraft`` deck) — and swaps only the
battle net's OWN draft partner across a quality ladder
(random -> greedy -> balanced -> edraft -> ldraft). The spread of the net's win
rate across that ladder IS its deck affinity: a deck-robust net barely moves, a
deck-dependent one collapses with a bad draft.

Runs both the new RoR (e36 gen7) and the prior RoR (e29slim) so the affinity
CURVES are directly comparable — is the self-play net more or less
deck-dependent than its predecessor? This is a cleaner cut than the 2026-06-25
"PPO x draft sweep" (which varied the draft on both sides and conflated draft
quality with the battle matchup); here the opponent is held constant so the read
is pure battle-side deck sensitivity.

Metric = the NET's win rate over the fixed oracle (higher = better). Same game
seeds per draft across the two nets (CRN), so the e36-vs-e29slim delta is paired.

    python scripts/e36_draft_affinity.py --pairs 125 --workers 16
    python scripts/e36_draft_affinity.py --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from locma.stats.intervals import wilson_ci

# Battle-net paths; each is paired with every draft below.
NETS = {
    "e36_gen7": "depot:e36/e36_gen7.zip",
    "e29slim": "depot:e29slim/e29slim_s0.zip",
}
# Draft quality ladder, worst -> best (hypothesis).
DRAFTS = ["random", "greedy", "balanced", "edraft", "ldraft"]
# Fixed opponent: fair determinized MCTS at the near-parity budget (E36 ladder
# medium rung), with a CONSTANT strong ldraft deck. Held identical every cell.
ORACLE = "dmcts:15,60,0,3,depot:ldraft/ldraft_s0.zip"
# Per-draft seed base; both nets share it within a draft (CRN).
DRAFT_SEED = {d: 73_000_000 + i * 1_000_000 for i, d in enumerate(DRAFTS)}

_CACHE: dict = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _make_draft(key: str):
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.policies.drafts import (  # noqa: PLC0415
        BalancedDraftPolicy,
        DistilledDraftPolicy,
        GreedyDraftPolicy,
        RandomDraftPolicy,
    )
    from locma.policies.ppo import MaskablePPODraftPolicy  # noqa: PLC0415

    if key == "random":
        return RandomDraftPolicy(seed=0)
    if key == "greedy":
        return GreedyDraftPolicy()
    if key == "balanced":
        return BalancedDraftPolicy()
    if key == "edraft":
        return DistilledDraftPolicy.load(resolve_path("depot:edraft/e20-elicit-fit.json"))
    if key == "ldraft":
        return MaskablePPODraftPolicy(model_path=resolve_path("depot:ldraft/ldraft_s0.zip"))
    raise ValueError(f"unknown draft {key}")


def _cell(net_path: str, draft_key: str, seed: int, pairs: int) -> tuple[int, int]:
    """One seed block: the battle net (net_path) + draft_key vs the fixed oracle."""
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.ppo import MaskablePPOBattlePolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    if net_path not in _CACHE:
        _CACHE[net_path] = MaskablePPOBattlePolicy(model_path=resolve_path(net_path))
    dkey = f"draft::{draft_key}"
    if dkey not in _CACHE:
        _CACHE[dkey] = _make_draft(draft_key)
    if ORACLE not in _CACHE:
        _CACHE[ORACLE] = make_policy(ORACLE)
    net = Composer(_CACHE[net_path], _CACHE[dkey], name=f"{net_path}+{draft_key}")
    res = run_match(net, _CACHE[ORACLE], games=pairs, seed=seed)
    return res.wins_a, res.games


def _noop() -> None:
    try:
        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        init_eval_worker()
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=125, help="seed pairs/cell (n = 2*pairs)")
    ap.add_argument("--block-pairs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="3 pairs, serial")
    ap.add_argument("--out", default="runs/e36/draft_affinity.json")
    args = ap.parse_args()

    pairs = 3 if args.smoke else args.pairs
    workers = 1 if args.smoke else args.workers
    block = args.block_pairs

    units = []  # (net_label, net_path, draft, seed, n)
    for draft in DRAFTS:
        for nl, npath in NETS.items():
            off = 0
            while off < pairs:
                n = min(block, pairs - off)
                units.append((nl, npath, draft, DRAFT_SEED[draft] + off, n))
                off += n

    print(f"E36 draft affinity — {utc_now()}  {len(units)} blocks on {workers} workers")
    print(f"  oracle (fixed): {ORACLE}")
    agg: dict = {(nl, d): [0, 0] for d in DRAFTS for nl in NETS}
    t0 = time.perf_counter()

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_noop) as ex:
            futs = {ex.submit(_cell, u[1], u[2], u[3], u[4]): (u[0], u[2], u[3]) for u in units}
            for f in as_completed(futs):
                nl, d, seed = futs[f]
                w, g = f.result()
                agg[(nl, d)][0] += w
                agg[(nl, d)][1] += g
                print(f"  [{d:9s}] {nl:9s} seed {seed}: {w}/{g}", flush=True)
    else:
        for u in units:
            w, g = _cell(u[1], u[2], u[3], u[4])
            agg[(u[0], u[2])][0] += w
            agg[(u[0], u[2])][1] += g

    table: dict = {}
    for d in DRAFTS:
        row = {}
        for nl in NETS:
            w, g = agg[(nl, d)]
            lo, hi = wilson_ci(w, g)
            row[nl] = {"net_wr": round(w / g, 4), "ci": [round(lo, 4), round(hi, 4)], "n": g}
        row["delta_e36_minus_e29slim"] = round(
            row["e36_gen7"]["net_wr"] - row["e29slim"]["net_wr"], 4
        )
        table[d] = row

    # affinity spread per net = best draft WR - worst draft WR (how much deck matters)
    spread = {}
    for nl in NETS:
        wrs = [table[d][nl]["net_wr"] for d in DRAFTS]
        spread[nl] = {"min": min(wrs), "max": max(wrs), "spread": round(max(wrs) - min(wrs), 4)}

    payload = {
        "generated": utc_now(),
        "oracle_fixed": ORACLE,
        "nets": NETS,
        "drafts": DRAFTS,
        "table": table,
        "affinity_spread": spread,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print("\n===== E36 draft affinity — net WR vs fixed dmcts:15,60(ldraft) oracle =====")
    print(f"{'draft':10s} {'e29slim':>16s} {'e36_gen7':>16s} {'e36-e29slim':>12s}")
    for d in DRAFTS:
        a, b = table[d]["e29slim"], table[d]["e36_gen7"]
        print(
            f"{d:10s} {a['net_wr']:.3f} [{a['ci'][0]:.2f},{a['ci'][1]:.2f}] "
            f"{b['net_wr']:.3f} [{b['ci'][0]:.2f},{b['ci'][1]:.2f}] "
            f"{table[d]['delta_e36_minus_e29slim']:+.3f}"
        )
    print("\naffinity spread (best-draft WR - worst-draft WR; smaller = more deck-robust):")
    for nl in NETS:
        s = spread[nl]
        print(f"  {nl:9s} {s['min']:.3f} .. {s['max']:.3f}  spread {s['spread']:.3f}")
    print(f"\nwrote {args.out}  ({payload['seconds']}s)")


if __name__ == "__main__":
    main()
