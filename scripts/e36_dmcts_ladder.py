"""E36 vs a fair-search oracle ladder: how the prior RoR (e29slim) and the new
RoR (e36 gen7) hold up as a determinized-MCTS opponent gets stronger.

Oracle = ``dmcts`` (determinized, NON-cheating multi-turn MCTS, K=15 worlds),
scaled over three iteration budgets (difficulty = K*I total sims). Draft is
matched to ``depot:ldraft`` on BOTH sides so the comparison isolates battle
play, not drafting. Each rung uses common random numbers across the two nets
(same game seeds), so the e36 - e29slim delta is a paired read.

Reported per (net, rung): the NET's win rate over the oracle (higher = the net
beats the search harder) with Wilson 95% CI, plus the delta vs the ``e29slim``
anchor when both are measured.

Also the RULER RE-PIN harness (2026-07-24): the E36 primary gate
(``rbeam:shared`` WR) is saturated at parity (3-seed pooled 0.509 [.481,.537]),
so it can no longer measure progress. The ``hard_2250sim`` rung is the
unsaturated replacement — gen7 sits at 0.390 there, with headroom in both
directions. ``--nets`` selects any subset of NETS (all three PFSP parity
endpoints are registered), and ``--pool-nets`` reports one pooled Wilson CI
across the seed replicates so the read is directly comparable to the pooled
parity number it replaces.

    python scripts/e36_dmcts_ladder.py --pairs 200 --workers 16   # original 2-net run
    python scripts/e36_dmcts_ladder.py --smoke                    # 3 pairs, serial
    # ruler re-pin: 3-seed endpoints + anchor on the hard rung only
    python scripts/e36_dmcts_ladder.py --rungs hard_2250sim \
        --nets e29slim,e36_gen7,e36_m1_gen7,e36_s22_gen7 \
        --pool-nets e36_gen7,e36_m1_gen7,e36_s22_gen7 --workers 8 \
        --out runs/e38/repin_hard.json
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

# Reactive nets, single-net, matched draft. The first two are the original
# ladder's default pair (prior RoR anchor + the promoted x86 gen7); the two
# extra PFSP endpoints are the M1 and s22 seed replicates that make up the
# 3-seed parity pool, registered here for the ruler re-pin.
_E36_TRIO = "depot:e36/e36_gen7.zip|depot:e36m1/e36_m1_gen7.zip|depot:e36s22/e36_s22_gen7.zip"
_E29_TRIO = "depot:e29slim/e29slim_s0.zip|depot:e29slim/e29slim_s1.zip|depot:e29slim/e29slim_s2.zip"

NETS = {
    "e29slim": f"ppo:depot:e29slim/e29slim_s0.zip,{LDRAFT}",
    "e36_gen7": f"ppo:depot:e36/e36_gen7.zip,{LDRAFT}",
    "e36_m1_gen7": f"ppo:depot:e36m1/e36_m1_gen7.zip,{LDRAFT}",
    "e36_s22_gen7": f"ppo:depot:e36s22/e36_s22_gen7.zip,{LDRAFT}",
    # E40 part B — the e36 3-seed ENSEMBLE, constructible with no training from
    # the three independent PFSP endpoints (x86 14M / m1 20M / s22 22M). This
    # closes baseline.md's standing caveat that "gen7 is one net, so there is no
    # `lppo:e36` guarded-ensemble variant" — written when only two chains existed.
    # NB these three differ in platform and n_envs as well as seed, so the trio
    # carries MORE diversity than the e29slim trio (E7/E8 say that helps an
    # ensemble; flagged, not assumed). `lppo:` adds the E26 lethal-guard lens.
    "e36_trio": f"ppo:{_E36_TRIO},{LDRAFT}",
    "e36_trio_guarded": f"lppo:{_E36_TRIO},{LDRAFT}",
    # Incumbent guarded-reactive recipe of record (E29-slim, 4 generations stale)
    # — the comparison the guarded promotion has to beat.
    "e29slim_trio_guarded": f"lppo:{_E29_TRIO},{LDRAFT}",
}
DEFAULT_NETS = ("e29slim", "e36_gen7")

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
    ap.add_argument(
        "--nets",
        default=",".join(DEFAULT_NETS),
        help=f"comma list of net labels from {sorted(NETS)}",
    )
    ap.add_argument(
        "--rungs", default=",".join(RUNGS), help=f"comma list of rung labels from {list(RUNGS)}"
    )
    ap.add_argument(
        "--pool-nets",
        default="",
        help="comma list of net labels to ALSO report as one pooled Wilson CI "
        "(seed replicates of the same recipe; e.g. the 3 PFSP parity endpoints)",
    )
    ap.add_argument("--out", default="runs/e36/dmcts_ladder.json")
    ap.add_argument(
        "--extra-net",
        action="append",
        default=[],
        metavar="LABEL=SPEC",
        help="register an additional net label (repeatable), e.g. "
        "--extra-net e42h_gen9=ppo:runs/e36_e42h_gen9.zip,depot:ldraft/ldraft_s0.zip "
        "so it can be named in --nets/--pool-nets without editing this file",
    )
    args = ap.parse_args()

    for item in args.extra_net:
        if "=" not in item:
            ap.error(f"--extra-net must be LABEL=SPEC, got {item!r}")
        label, spec = item.split("=", 1)
        label, spec = label.strip(), spec.strip()
        if not label or not spec:
            ap.error(f"--extra-net must be LABEL=SPEC, got {item!r}")
        NETS[label] = spec

    pairs = 3 if args.smoke else args.pairs
    workers = 1 if args.smoke else args.workers
    block = args.block_pairs

    net_labels = [s.strip() for s in args.nets.split(",") if s.strip()]
    rung_labels = [s.strip() for s in args.rungs.split(",") if s.strip()]
    pool_labels = [s.strip() for s in args.pool_nets.split(",") if s.strip()]
    for label in net_labels + pool_labels:
        if label not in NETS:
            ap.error(f"unknown net label {label!r}; known: {sorted(NETS)}")
    for label in rung_labels:
        if label not in RUNGS:
            ap.error(f"unknown rung label {label!r}; known: {list(RUNGS)}")
    missing = [label for label in pool_labels if label not in net_labels]
    if missing:
        ap.error(f"--pool-nets labels must also be in --nets; missing: {missing}")
    nets = {label: NETS[label] for label in net_labels}
    rungs = {label: RUNGS[label] for label in rung_labels}

    units = []  # (net_label, net_spec, rung_label, oracle_spec, seed, n)
    for rung, oracle in rungs.items():
        for net_label, net_spec in nets.items():
            off = 0
            while off < pairs:
                n = min(block, pairs - off)
                units.append((net_label, net_spec, rung, oracle, RUNG_SEED[rung] + off, n))
                off += n

    print(f"E36 dmcts-oracle ladder — {utc_now()}  {len(units)} blocks on {workers} workers")
    print(f"  nets={net_labels}  rungs={rung_labels}  pool={pool_labels or '—'}")
    agg: dict = {(nl, rg): [0, 0] for rg in rungs for nl in nets}  # wins, games
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
    for rg in rungs:
        row = {}
        for nl in nets:
            w, g = agg[(nl, rg)]
            lo, hi = wilson_ci(w, g)
            row[nl] = {
                "net_wr_vs_oracle": round(w / g, 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "n": g,
            }
        if "e36_gen7" in nets and "e29slim" in nets:
            row["delta_e36_minus_e29slim"] = round(
                row["e36_gen7"]["net_wr_vs_oracle"] - row["e29slim"]["net_wr_vs_oracle"], 4
            )
        if len(pool_labels) > 1:
            pw = sum(agg[(nl, rg)][0] for nl in pool_labels)
            pg = sum(agg[(nl, rg)][1] for nl in pool_labels)
            lo, hi = wilson_ci(pw, pg)
            row["pooled"] = {
                "members": pool_labels,
                "net_wr_vs_oracle": round(pw / pg, 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "n": pg,
            }
        ladder[rg] = row

    payload = {
        "generated": utc_now(),
        "oracle": "dmcts (fair, non-cheating), K=15 worlds, matched ldraft draft",
        "nets": nets,
        "rungs": rungs,
        "pool_nets": pool_labels,
        "ladder": ladder,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print(
        "\n============ E36 vs dmcts-oracle ladder (net WR over oracle, higher=better) ============"
    )
    for rg in rungs:
        sims = int(rungs[rg].split(",")[0].split(":")[1]) * int(rungs[rg].split(",")[1])
        print(f"\n{rg}  ({sims} sims)")
        keys = [*net_labels, *(["pooled"] if len(pool_labels) > 1 else [])]
        for nl in keys:
            c = ladder[rg][nl]
            tag = f"pooled({len(pool_labels)} seeds)" if nl == "pooled" else nl
            print(
                f"  {tag:22s} {c['net_wr_vs_oracle']:.4f} "
                f"[{c['ci'][0]:.3f},{c['ci'][1]:.3f}]  (n={c['n']})"
            )
        if "delta_e36_minus_e29slim" in ladder[rg]:
            print(f"  {'e36 - e29slim':22s} {ladder[rg]['delta_e36_minus_e29slim']:+.4f}")
    print(f"\nwrote {args.out}  ({payload['seconds']}s)")


def _noop() -> None:  # pool initializer: quiet torch threads via the shared helper if present
    try:
        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        init_eval_worker()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
