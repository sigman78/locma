"""E40 non-regression: does the re-pinned `rbeam` evaluator hold up on the POOL ruler?

E24 promoted `rbeam` on a head-to-head ruler and then explicitly checked it did not
regress on the avg-hard3 pool (`rbeam:shared` scored 0.983 there). E40-A re-pins the
evaluator net, so it owes the same check. The pool saturates near 1.0 for every
strong config, so this is a non-regression gate, NOT a strength claim.

Also scores the guarded-reactive promotion candidate on the same pool for the same
reason (the incumbent `lppo:e29slim` trio holds 0.934 there).

Must live in a FILE, not a heredoc: ProcessPoolExecutor uses spawn on macOS and
re-imports __main__ from its path, so a `python -` script dies with
`FileNotFoundError: <stdin>`.

    python scripts/e40_pool_check.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from locma.harness.parallel import init_eval_worker
from locma.stats.intervals import wilson_ci

LDRAFT = "depot:ldraft/ldraft_s0.zip"
_E36_TRIO = "depot:e36/e36_gen7.zip|depot:e36m1/e36_m1_gen7.zip|depot:e36s22/e36_s22_gen7.zip"

CANDIDATES = {
    # search RoR candidates (E40-A)
    "rbeam_gen7": f"rbeam:depot:e36/e36_gen7.zip,8,20,4,4,{LDRAFT}",
    "rbeam_e36trio": f"rbeam:{_E36_TRIO},8,20,4,4,{LDRAFT}",
    # guarded-reactive promotion candidate (E40-B)
    "lppo_e36trio": f"lppo:{_E36_TRIO},{LDRAFT}",
}
HARD3 = ("scripted", "max-guard", "max-attack")
SEED0 = 84_000_000

# Published references on this ruler, for the non-regression read.
REFERENCE = {"rbeam:shared (E24)": 0.983, "lppo:e29slim trio (E29-slim)": 0.934}


def _cell(cand_spec: str, opp: str, seed: int, pairs: int) -> tuple[str, int, int]:
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    res = run_match(make_policy(cand_spec), make_policy(opp), games=pairs, seed=seed)
    return opp, res.wins_a, res.games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=100, help="seed pairs/opponent (n = 2*pairs)")
    ap.add_argument("--block-pairs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cands", default=",".join(CANDIDATES))
    ap.add_argument("--out", default="runs/e40/pool_check.json")
    args = ap.parse_args()

    labels = [s.strip() for s in args.cands.split(",") if s.strip()]
    for label in labels:
        if label not in CANDIDATES:
            ap.error(f"unknown candidate {label!r}; known: {list(CANDIDATES)}")

    out: dict = {"reference": REFERENCE, "candidates": {}}
    t0 = time.perf_counter()
    for ci, label in enumerate(labels):
        spec = CANDIDATES[label]
        units = [
            (spec, opp, SEED0 + ci * 10_000 + oi * 1000 + off, min(args.block_pairs, args.pairs))
            for oi, opp in enumerate(HARD3)
            for off in range(0, args.pairs, args.block_pairs)
        ]
        agg = {opp: [0, 0] for opp in HARD3}
        print(f"[{label}] {len(units)} blocks on {args.workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_eval_worker) as ex:
            for f in as_completed([ex.submit(_cell, *u) for u in units]):
                opp, w, g = f.result()
                agg[opp][0] += w
                agg[opp][1] += g
        per, rates = {}, []
        for opp in HARD3:
            w, g = agg[opp]
            lo, hi = wilson_ci(w, g)
            per[opp] = {"wr": round(w / g, 4), "ci": [round(lo, 4), round(hi, 4)], "n": g}
            rates.append(w / g)
            print(f"  {opp:12s} {w / g:.4f} [{lo:.3f},{hi:.3f}] (n={g})", flush=True)
        avg = sum(rates) / len(rates)
        out["candidates"][label] = {"spec": spec, "per_opp": per, "avg_hard3": round(avg, 4)}
        print(f"  avg-hard3    {avg:.4f}", flush=True)

    out["seconds"] = round(time.perf_counter() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n======== E40 POOL NON-REGRESSION (avg-hard3, saturated ruler) ========")
    for label in labels:
        print(f"{label:16s} {out['candidates'][label]['avg_hard3']:.4f}")
    for k, v in REFERENCE.items():
        print(f"{'reference':16s} {v:.4f}  <- {k}")
    print(f"\nwrote {args.out}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
