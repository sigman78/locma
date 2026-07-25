"""E39 Gate 0: is the net's OWN lookahead worth following?

Pre-registered 2026-07-24. E38 closed with a sharp mechanism: what the reactive
net lacks is not information about immediate consequences (handing it the engine's
own per-action answer moved plan agreement +0.021) but the MULTI-TURN VALUATION of
those consequences. The E33 reply-aware instrument then localized that inside the
net: gen7 disagrees with its OWN value head under one reply ply on **40.6%** of
turn-openings. The oracle there is the same net, so the information is present — in
the value head — and the policy head is not acting on it.

The obvious large follow-up is to amortize that deeper self into the policy head
(an auxiliary search-target loss inside the PFSP loop). It rests on an untested
premise: **that following the deeper plan actually WINS MORE GAMES.** Nobody has
measured that for gen7 — E33 computes the plan and compares it, but never plays
it. And there is a real prior it may not: E32 Phase 1 found e29slim was a WORSE
`rbeam` evaluator than `shared` (0.335), localized to the value tower.

So: play the net's own lookahead against the net's bare policy head, matched draft,
same net on both sides. The search side's win rate is the value of following your
own critic.

Pre-registered reads (stage 1, `own_search_wr` over the bare net):

  >= ~0.60   the value head genuinely knows better; the 0.406 disagreement is
             costly -> open E39-align (search-derived plan target as an AUXILIARY
             loss inside PPO, on-policy, full-net — the three things that separate
             it from the closed E4v2/E15 imitation arms).
  ~ parity   the value head adds nothing on this net (consistent with E32) ->
             the bottleneck is critic QUALITY, so go at the orphaned unfrozen
             ranking-loss rerun instead (E15's wall was explicitly
             frozen-extractor and was never retried).
  < 0.50     the value head is WORSE than the policy head -> PFSP optimized the
             policy at the critic's expense; any search lever needs a dedicated
             critic first.

Stage 2 (gated on stage 1): put the winning own-search config on the re-pinned
primary ruler `dmcts:15,150`, where the bare net sits at 0.390 (x86 gen7) — does
the internal gain convert against real opposition?

    python scripts/e39_gate0.py --stage 1 --pairs 200 --workers 8
    python scripts/e39_gate0.py --stage 2 --pairs 200 --workers 8
    python scripts/e39_gate0.py --smoke
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

# x86 gen7 is the pure-reactive recipe of record; m1 gen7 is the endpoint the
# 0.406 root-disagree number was measured on (baseline.md calls the two a dead
# heat), so it runs as a cross-check that the gate is not a per-endpoint artifact.
NETS = {
    "x86_gen7": "depot:e36/e36_gen7.zip",
    "m1_gen7": "depot:e36m1/e36_m1_gen7.zip",
}


# Own-search configs: the SAME net is the evaluator, so the only difference from
# the baseline is the search wrapped around it. vbeam = one own-turn ply; rbeam =
# plus one genuine opponent-reply ply (the 8,20,4,4 search RoR depth).
def own_vbeam(net: str) -> str:
    return f"vbeam:{net},8,20,{LDRAFT}"


def own_rbeam(net: str) -> str:
    return f"rbeam:{net},8,20,4,4,{LDRAFT}"


def bare(net: str) -> str:
    return f"ppo:{net},{LDRAFT}"


DMCTS_HARD = f"dmcts:15,150,0,3,{LDRAFT}"  # the re-pinned primary ruler

# Seed bases, clear of 5/12/14/20/22/24/26/28/30/40/60/70-72M already spent.
SEED_STAGE1 = 80_000_000
SEED_STAGE2 = 81_000_000

_CACHE: dict = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _cell(a_spec: str, b_spec: str, seed: int, pairs: int) -> tuple[int, int, float]:
    """One seed block of run_match(a, b): returns (a_wins, games, seconds)."""
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


def run_cells(cells: list[tuple[str, str, str]], pairs: int, block: int, workers: int) -> dict:
    """cells = [(label, a_spec, b_spec)]; a's win rate is the reported number."""
    units = []
    for label, a, b in cells:
        seed0 = cells_seed[label]
        off = 0
        while off < pairs:
            n = min(block, pairs - off)
            units.append((label, a, b, seed0 + off, n))
            off += n
    print(f"  {len(units)} blocks on {workers} workers", flush=True)

    agg: dict[str, list[float]] = {label: [0, 0, 0.0] for label, _a, _b in cells}
    if workers > 1 and len(units) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_noop) as ex:
            futs = {ex.submit(_cell, u[1], u[2], u[3], u[4]): (u[0], u[3]) for u in units}
            for f in as_completed(futs):
                label, seed = futs[f]
                w, g, sec = f.result()
                agg[label][0] += w
                agg[label][1] += g
                agg[label][2] += sec
                print(f"    [{label:22s}] seed {seed}: {w}/{g} ({sec:.0f}s)", flush=True)
    else:
        for u in units:
            w, g, sec = _cell(u[1], u[2], u[3], u[4])
            agg[u[0]][0] += w
            agg[u[0]][1] += g
            agg[u[0]][2] += sec

    out = {}
    for label, a, b in cells:
        w, g, sec = agg[label]
        lo, hi = wilson_ci(int(w), int(g))
        out[label] = {
            "a": a,
            "b": b,
            "a_wr": round(w / g, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "n": int(g),
            "s_per_game": round(sec / g, 2) if g else None,
        }
    return out


cells_seed: dict[str, int] = {}


def verdict_stage1(wr: float, ci: list[float]) -> str:
    if ci[0] >= 0.55:
        return "own lookahead WINS decisively -> open E39-align (amortize it)"
    if ci[1] < 0.50:
        return "own lookahead LOSES -> the value head is worse than the policy head"
    if ci[0] > 0.50:
        return "own lookahead wins but modestly -> judge vs the 0.60 bar"
    return (
        "PARITY -> value head adds nothing on this net; go at critic QUALITY (E15 unfrozen rerun)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--pairs", type=int, default=200, help="seed pairs/cell (n = 2*pairs)")
    ap.add_argument("--block-pairs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="3 pairs, serial")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pairs = 3 if args.smoke else args.pairs
    workers = 1 if args.smoke else args.workers
    out = args.out or f"runs/e39/gate0_stage{args.stage}.json"

    if args.stage == 1:
        # Own-search vs the SAME net's bare policy head. Distinct seed base per
        # cell so the cells are independent samples, not the same games reused.
        cells = [
            ("vbeam_x86", own_vbeam(NETS["x86_gen7"]), bare(NETS["x86_gen7"])),
            ("rbeam_x86", own_rbeam(NETS["x86_gen7"]), bare(NETS["x86_gen7"])),
            ("vbeam_m1", own_vbeam(NETS["m1_gen7"]), bare(NETS["m1_gen7"])),
        ]
        for i, (label, _a, _b) in enumerate(cells):
            cells_seed[label] = SEED_STAGE1 + i * 1000
    else:
        # Own-search vs the re-pinned primary ruler. Baselines for comparison:
        # the BARE nets scored 0.390 (x86) / 0.395 (m1) here (runs/e38/repin_dmcts.json).
        cells = [
            ("vbeam_vs_dmcts", own_vbeam(NETS["x86_gen7"]), DMCTS_HARD),
            ("rbeam_vs_dmcts", own_rbeam(NETS["x86_gen7"]), DMCTS_HARD),
        ]
        for i, (label, _a, _b) in enumerate(cells):
            cells_seed[label] = SEED_STAGE2 + i * 1000

    print(f"E39 Gate 0 stage {args.stage} — {utc_now()}  {len(cells)} cells x {pairs} pairs")
    t0 = time.perf_counter()
    res = run_cells(cells, pairs, args.block_pairs, workers)

    payload = {
        "generated": utc_now(),
        "stage": args.stage,
        "question": (
            "does following the net's OWN lookahead beat its bare policy head?"
            if args.stage == 1
            else "does the own-lookahead gain convert on the re-pinned dmcts:15,150 ruler?"
        ),
        "seeds": dict(cells_seed),
        "cells": res,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if args.stage == 1:
        payload["verdicts"] = {k: verdict_stage1(v["a_wr"], v["ci"]) for k, v in res.items()}
    else:
        # bare-net reference cells from the ruler re-pin, for a direct delta
        payload["bare_net_reference"] = {"x86_gen7": 0.3900, "m1_gen7": 0.3950}
        payload["delta_vs_bare"] = {k: round(v["a_wr"] - 0.3900, 4) for k, v in res.items()}

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))

    print(f"\n============ E39 GATE 0 — STAGE {args.stage} ============")
    for label, v in res.items():
        print(
            f"{label:18s} {v['a_wr']:.4f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] "
            f"(n={v['n']}, {v['s_per_game']}s/game)"
        )
        if args.stage == 1:
            print(f"{'':18s} -> {payload['verdicts'][label]}")
        else:
            print(f"{'':18s} -> vs bare net 0.3900: {payload['delta_vs_bare'][label]:+.4f}")
    print(f"\nwrote {out}  ({payload['seconds']}s)")


if __name__ == "__main__":
    main()
