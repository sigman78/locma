"""E42 guard-rail G: boardkeep exploit win-rate vs the two Gate 1 arms.

Same protocol as ``runs/e36/gen7_boardkeep_guard.json`` (the x86 gen7 anchor:
boardkeep 0.1675 [0.152, 0.185], 2000 mirrored games, seed base 5M) so the
arms are read against a published number on common random numbers.

Usage:
    .venv/Scripts/python scripts/e42_guard.py [--pairs 1000] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from locma.stats.intervals import wilson_ci

LDRAFT = "depot:ldraft/ldraft_s0.zip"
ARMS = {
    "e42h_gen9": f"ppo:runs/e36_e42h_gen9.zip,{LDRAFT}",
    "e42c_gen9": f"ppo:runs/e36_e42c_gen9.zip,{LDRAFT}",
}
SEED0 = 5_000_000
_CACHE: dict = {}


def _block(defender: str, seed: int, pairs: int) -> tuple[int, int]:
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    for spec in ("boardkeep", defender):
        if spec not in _CACHE:
            _CACHE[spec] = make_policy(spec)
    res = run_match(_CACHE["boardkeep"], _CACHE[defender], games=pairs, seed=seed)
    return res.wins_a, res.games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--block-pairs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/e42/guard.json")
    args = ap.parse_args()

    t0 = time.time()
    out: dict = {"protocol": "boardkeep vs arm, mirrored, seed base 5M", "arms": {}}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for label, spec in ARMS.items():
            futs = [
                ex.submit(_block, spec, SEED0 + off, min(args.block_pairs, args.pairs - off))
                for off in range(0, args.pairs, args.block_pairs)
            ]
            w = g = 0
            for f in futs:
                bw, bg = f.result()
                w += bw
                g += bg
            lo, hi = wilson_ci(w, g)
            out["arms"][label] = {"boardkeep_wr": w / g, "wilson_ci": [lo, hi], "games": g}
            print(f"  {label:12s} boardkeep WR {w / g:.4f} [{lo:.3f},{hi:.3f}] (n={g})", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} ({out['seconds']}s)")


if __name__ == "__main__":
    main()
