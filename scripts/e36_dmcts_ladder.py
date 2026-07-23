"""E36 vs a fair-search oracle ladder: how the prior RoR (e29slim) and the new
RoR (e36 gen7) hold up as a determinized-MCTS opponent gets stronger.

Oracle = ``dmcts`` (determinized, NON-cheating multi-turn MCTS, K=15 worlds),
scaled over three iteration budgets (difficulty = K*I total sims). Draft is
matched to ``depot:ldraft`` on BOTH sides so the comparison isolates battle
play, not drafting. Each rung uses common random numbers across the two nets
(same game seeds), so the e36 - e29slim delta is a paired read.

Reported per (net, rung): the NET's win rate over the oracle (higher = the net
beats the search harder) with Wilson 95% CI, plus the e36-vs-e29slim delta.

    python scripts/e36_dmcts_ladder.py --pairs 200 --workers 16
    python scripts/e36_dmcts_ladder.py --smoke        # 3 pairs, serial
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

# The two reactive recipes of record, single-net, matched draft.
NETS = {
    "e29slim": f"ppo:depot:e29slim/e29slim_s0.zip,{LDRAFT}",
    "e36_gen7": f"ppo:depot:e36/e36_gen7.zip,{LDRAFT}",
}

# Fair determinized-MCTS oracle, K=15 worlds, matched ldraft draft (5th param),
# scaled by iterations/world. Difficulty = 15 * I total simulations.
RUNGS = {
    "easy_300sim": f"dmcts:15,20,0,3,{LDRAFT}",
    "med_900sim": f"dmcts:15,60,0,3,{LDRAFT}",
    "hard_2250sim": f"dmcts:15,150,0,3,{LDRAFT}",
}
# Distinct per-rung seed base; both nets share it within a rung (CRN).
RUNG_SEED = {"easy_300sim": 70_000_000, "med_900sim": 71_000_000, "hard_2250sim": 72_000_000}

_CACHE: dict = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _cell(net_spec: str, oracle_spec: str, seed: int, pairs: int) -> tuple[int, int]:
    """One seed block of run_match(net, oracle): returns (net_wins, games)."""
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    for spec in (net_spec, oracle_spec):
        if spec not in _CACHE:
            _CACHE[spec] = make_policy(spec)
    res = run_match(_CACHE[net_spec], _CACHE[oracle_spec], games=pairs, seed=seed)
    return res.wins_a, res.games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=200, help="seed pairs/cell (n = 2*pairs)")
    ap.add_argument("--block-pairs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="3 pairs, serial")
    ap.add_argument("--out", default="runs/e36/dmcts_ladder.json")
    args = ap.parse_args()

    pairs = 3 if args.smoke else args.pairs
    workers = 1 if args.smoke else args.workers
    block = args.block_pairs

    units = []  # (net_label, net_spec, rung_label, oracle_spec, seed, n)
    for rung, oracle in RUNGS.items():
        for net_label, net_spec in NETS.items():
            off = 0
            while off < pairs:
                n = min(block, pairs - off)
                units.append((net_label, net_spec, rung, oracle, RUNG_SEED[rung] + off, n))
                off += n

    print(f"E36 dmcts-oracle ladder — {utc_now()}  {len(units)} blocks on {workers} workers")
    agg: dict = {(nl, rg): [0, 0] for rg in RUNGS for nl in NETS}  # wins, games
    t0 = time.perf_counter()

    def absorb(key, wins, games):
        agg[key][0] += wins
        agg[key][1] += games

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_noop) as ex:
            futs = {ex.submit(_cell, u[1], u[3], u[4], u[5]): (u[0], u[2], u[4]) for u in units}
            for f in as_completed(futs):
                nl, rg, seed = futs[f]
                w, g = f.result()
                absorb((nl, rg), w, g)
                print(f"  [{rg:12s}] {nl:9s} seed {seed}: {w}/{g}", flush=True)
    else:
        for u in units:
            w, g = _cell(u[1], u[3], u[4], u[5])
            absorb((u[0], u[2]), w, g)

    ladder: dict = {}
    for rg in RUNGS:
        row = {}
        for nl in NETS:
            w, g = agg[(nl, rg)]
            lo, hi = wilson_ci(w, g)
            row[nl] = {
                "net_wr_vs_oracle": round(w / g, 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "n": g,
            }
        row["delta_e36_minus_e29slim"] = round(
            row["e36_gen7"]["net_wr_vs_oracle"] - row["e29slim"]["net_wr_vs_oracle"], 4
        )
        ladder[rg] = row

    payload = {
        "generated": utc_now(),
        "oracle": "dmcts (fair, non-cheating), K=15 worlds, matched ldraft draft",
        "nets": NETS,
        "rungs": RUNGS,
        "ladder": ladder,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(
        "\n============ E36 vs dmcts-oracle ladder (net WR over oracle, higher=better) ============"
    )
    print(f"{'rung':14s} {'sims':>6s} {'e29slim':>18s} {'e36_gen7':>18s} {'e36-e29slim':>12s}")
    for rg in RUNGS:
        sims = int(RUNGS[rg].split(",")[0].split(":")[1]) * int(RUNGS[rg].split(",")[1])
        a, b = ladder[rg]["e29slim"], ladder[rg]["e36_gen7"]
        print(
            f"{rg:14s} {sims:>6d} "
            f"{a['net_wr_vs_oracle']:.3f} [{a['ci'][0]:.2f},{a['ci'][1]:.2f}] "
            f"{b['net_wr_vs_oracle']:.3f} [{b['ci'][0]:.2f},{b['ci'][1]:.2f}] "
            f"{ladder[rg]['delta_e36_minus_e29slim']:+.3f}"
        )
    print(f"\nwrote {args.out}  ({payload['seconds']}s)")


def _noop() -> None:  # pool initializer: quiet torch threads via the shared helper if present
    try:
        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        init_eval_worker()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
