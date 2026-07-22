"""E36 scale-check gate evals — run the two Gate-0 rulers on the PFSP chain.

The Gate-0 signal test (2 generations) moved both rulers: avg-hard3 0.919->0.956
and the held-out fair-search gap 0.807->0.703. The scale check (gen2-4, driver
``scripts/e36_pfsp.py --resume``) asks whether the gain GROWS. This script scores
the ladder on the identical rulers so the numbers are directly comparable:

  1. **search gap** (the trustworthy read — avg-hard3 pool-saturates): the fair
     play-time-search recipe of record ``rbeam:shared`` (3-net ensemble, the
     8,20,4,4 RoR config) head-to-head vs the pure net ``ppo:<net>,ldraft``.
     SEED base 30M, 200 seed pairs (n=400 games), Wilson CI. This is the
     candidate's win rate over the pure net, so LOWER = the pure net is harder
     for fair search to beat (a smaller gap). The e29slim control must reproduce
     Gate-0's 0.807 (== Phase-3's 0.808) — methodology validation.

  2. **avg-hard3**: pure-net mean win rate vs the three hard scripted baselines
     ``(scripted, max-guard, max-attack)``, 150 games/opp, seed 40M. HIGHER is
     better; near-saturated at the top of the ladder.

Run the ladder so the control anchors the read:

    .venv/Scripts/python scripts/e36_gate_gen4.py --smoke --nets e29slim,gen1
    .venv/Scripts/python scripts/e36_gate_gen4.py --workers 4      # full, all nets
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

E29_TRIO = "depot:e29slim/e29slim_s0.zip|depot:e29slim/e29slim_s1.zip|depot:e29slim/e29slim_s2.zip"
SHARED = "depot:shared/shared_s0.zip|depot:shared/shared_s1.zip|depot:shared/shared_s2.zip"
LDRAFT = "depot:ldraft/ldraft_s0.zip"

# The fair play-time-search recipe of record (evaluator = shared ensemble).
RBEAM_SHARED = f"rbeam:{SHARED},8,20,4,4,{LDRAFT}"
SEARCH_SEED0 = 30_000_000  # the E22/E23/E24/E32 head-to-head ruler seed base
HARD3_SEED = 40_000_000  # Gate-0 avg-hard3 seed
HARD3_OPPS = ("scripted", "max-guard", "max-attack")

# label -> pure-net spec. The control is the e29slim trio (reproduces Gate-0's
# 0.807); gen1/gen4 are the single self-play nets. Both search arms compare the
# same rbeam:shared candidate against these.
LADDER = {
    "e29slim": f"ppo:{E29_TRIO},{LDRAFT}",
    "gen1": f"ppo:runs/e36_gen1.zip,{LDRAFT}",
    "gen4": f"ppo:runs/e36_gen4.zip,{LDRAFT}",
    "gen5": f"ppo:runs/e36_gen5.zip,{LDRAFT}",
    "gen6": f"ppo:runs/e36_gen6.zip,{LDRAFT}",
    "gen7": f"ppo:runs/e36_gen7.zip,{LDRAFT}",
    # M1 replication chain (depot:e36m1) — depot refs so the ladder runs on either box.
    **{
        f"m1_gen{g}": f"ppo:depot:e36m1/e36_m1_gen{g}.zip,{LDRAFT}"
        for g in range(8)
    },
}

_WORKER_POLICIES: dict[str, tuple[str, object]] = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _cached_policy(slot: str, spec: str):
    """Per-worker policy cache keyed by slot; rebuilds only when the spec changes."""
    entry = _WORKER_POLICIES.get(slot)
    if entry is not None and entry[0] == spec:
        return entry[1]
    if entry is not None:
        del _WORKER_POLICIES[slot]
        gc.collect()
    from locma.policies.registry import make_policy  # noqa: PLC0415

    _WORKER_POLICIES[slot] = (spec, make_policy(spec))
    return _WORKER_POLICIES[slot][1]


def _search_block(net_label: str, net_spec: str, seed: int, pairs: int) -> dict:
    """rbeam:shared (candidate) vs ppo:<net> (baseline): candidate WR = search gap."""
    from locma.harness.match import run_match  # noqa: PLC0415

    t0 = time.perf_counter()
    cand = _cached_policy("search_cand", RBEAM_SHARED)
    base = _cached_policy("search_base", net_spec)
    res = run_match(cand, base, games=pairs, seed=seed)
    return {
        "net": net_label,
        "seed": seed,
        "games": res.games,
        "candidate_wins": res.wins_a,
        "worker_seconds": round(time.perf_counter() - t0, 2),
    }


def _hard3_block(net_label: str, net_spec: str, opp: str, seed: int, games: int) -> dict:
    """Pure net vs one hard scripted baseline: net WR = win_rate_a."""
    from locma.harness.match import run_match  # noqa: PLC0415

    net = _cached_policy("hard3_net", net_spec)
    opp_pol = _cached_policy("hard3_opp", opp)
    res = run_match(net, opp_pol, games=games, seed=seed)
    return {"net": net_label, "opp": opp, "games": res.games, "wins": res.wins_a}


def run_search_gap(nets: dict[str, str], pairs: int, block_pairs: int, workers: int, log) -> dict:
    """Search-gap ruler for each net, blocked over seed pairs and parallelised."""
    units = []
    for label, spec in nets.items():
        offset = 0
        while offset < pairs:
            n = min(block_pairs, pairs - offset)
            units.append((label, spec, SEARCH_SEED0 + offset, n))
            offset += n
    log(
        f"[search] {len(nets)} nets x {pairs} pairs = {len(units)} blocks "
        f"({sum(u[3] for u in units) * 2} games) on {workers} workers"
    )

    agg: dict[str, list[int]] = {label: [0, 0] for label in nets}  # wins, games
    rows = []
    if workers > 1 and len(units) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_eval_worker) as ex:
            futs = [ex.submit(_search_block, *u) for u in units]
            for f in as_completed(futs):
                r = f.result()
                rows.append(r)
                agg[r["net"]][0] += r["candidate_wins"]
                agg[r["net"]][1] += r["games"]
                log(
                    f"  [search] {r['net']:9s} block seed {r['seed']}: "
                    f"{r['candidate_wins']}/{r['games']} ({r['worker_seconds']}s)"
                )
    else:
        for u in units:
            r = _search_block(*u)
            rows.append(r)
            agg[r["net"]][0] += r["candidate_wins"]
            agg[r["net"]][1] += r["games"]

    out = {}
    for label in nets:
        cw, g = agg[label]
        lo, hi = wilson_ci(cw, g)
        out[label] = {
            "search_wr_vs_net": round(cw / g, 4),
            "wilson_ci": [round(lo, 4), round(hi, 4)],
            "games": g,
        }
    return out


def run_hard3(nets: dict[str, str], games: int, workers: int, log) -> dict:
    """avg-hard3 ruler: pure-net mean WR vs the three hard scripted baselines."""
    units = [
        (label, spec, opp, HARD3_SEED + i * 1000, games)
        for label, spec in nets.items()
        for i, opp in enumerate(HARD3_OPPS)
    ]
    log(f"[hard3] {len(units)} matches ({games} games each) on {workers} workers")
    per: dict[str, dict[str, float]] = {label: {} for label in nets}
    if workers > 1 and len(units) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_eval_worker) as ex:
            futs = [ex.submit(_hard3_block, *u) for u in units]
            for f in as_completed(futs):
                r = f.result()
                per[r["net"]][r["opp"]] = r["wins"] / r["games"]
    else:
        for u in units:
            r = _hard3_block(*u)
            per[r["net"]][r["opp"]] = r["wins"] / r["games"]
    out = {}
    for label in nets:
        rates = [per[label][o] for o in HARD3_OPPS]
        out[label] = {
            "avg_hard3": round(sum(rates) / len(rates), 4),
            "per_opp": {o: round(per[label][o], 3) for o in HARD3_OPPS},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=200, help="search seed pairs/net (n = 2*pairs)")
    ap.add_argument("--block-pairs", type=int, default=25)
    ap.add_argument("--hard3-games", type=int, default=150, help="games per hard opponent")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nets", default="e29slim,gen1,gen4", help="comma list of ladder labels")
    ap.add_argument("--smoke", action="store_true", help="2 pairs, 4 hard3 games, serial")
    ap.add_argument(
        "--no-hard3",
        action="store_true",
        help="skip the avg-hard3 ruler (pool-saturated ~0.95 at the top of the ladder)",
    )
    ap.add_argument("--out", default="runs/e36/gen4_gate.json")
    args = ap.parse_args()

    labels = [s.strip() for s in args.nets.split(",") if s.strip()]
    nets = {label: LADDER[label] for label in labels}
    pairs = 2 if args.smoke else args.pairs
    hard3_games = 4 if args.smoke else args.hard3_games
    workers = 1 if args.smoke else args.workers

    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    log(f"E36 gen4 gate evals — {utc_now()}  nets={labels}")
    hard3 = None if args.no_hard3 else run_hard3(nets, hard3_games, workers, log)
    search = run_search_gap(nets, pairs, args.block_pairs, workers, log)

    payload = {
        "generated": utc_now(),
        "nets": nets,
        "hard3": hard3,
        "search_gap": search,
        "log": lines,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print("\n================ E36 GEN4 GATE LADDER ================")
    print(f"{'net':10s} {'avg-hard3':>10s} {'search-gap (rbeam:shared WR, lower=better)':>44s}")
    for label in labels:
        h = f"{hard3[label]['avg_hard3']:>10.3f}" if hard3 else f"{'—':>10s}"
        s = search[label]
        print(
            f"{label:10s} {h}   {s['search_wr_vs_net']:.3f} "
            f"CI[{s['wilson_ci'][0]:.3f},{s['wilson_ci'][1]:.3f}]  (n={s['games']})"
        )
    print(f"\nwrote {args.out}")
    print("Reproduction checks: e29slim avg-hard3 ~0.919, search-gap ~0.807; gen1 ~0.956 / ~0.703.")


if __name__ == "__main__":
    main()
