"""Zero-training trio driver: squeeze the existing nets without training anything.

Three studies, all on the standard paired ruler (avg-hard3, disjoint eval seeds):

  1. WIDTH SWEEP -- vbeam compute-scaling curve on the shared nets. w=8 is the
     0.890 recipe of record; w=4/16/32 tell us whether the planner is
     search-limited (curve still climbing) or evaluator-limited (flat). This
     gates whether the next investment is search depth or critic quality.
  2. CRITIC ENSEMBLE -- rank beam candidates with the mean of the three shared
     critics (vbeam:p0|p1|p2). Pure variance reduction on the sibling-ordering
     signal; zero training.
  3. RETRO-SCORING -- the draft-noise lesson (reactive nulls hiding better
     critics) applied to shelved checkpoints: selfplay-r2, az-net-0, sweep-C
     (the big net), vdst-ff (full-FT distill, the only distill arm whose critic
     changed), depot:cand1. Scored vs vbeam:depot:b0 for comparability with
     every E5/E7 delta.

Pre-registered retro gate: each retro arm runs a 10x10 pilot first and is
promoted to the full 40x25 ruler iff the pilot's ci_hi > 0 (not clearly
negative). Width and ensemble go straight to the full ruler.

Progress in runs/zero-overnight.log, results in runs/zero-summary.json.
Smoke mode: ZERO_SMOKE=1 -> 2x2 grids, separate summary/log files.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("ZERO_SMOKE") == "1"
WORKERS = 19
SEEDS = (0, 1, 2)
SUMMARY_PATH = "runs/zero-smoke.json" if SMOKE else "runs/zero-summary.json"
LOG_PATH = "runs/zero-smoke.log" if SMOKE else "runs/zero-overnight.log"
FULL = (2, 2) if SMOKE else (40, 25)  # (seeds, games_per_seed)
PILOT = (2, 2) if SMOKE else (10, 10)

SHARED = [f"depot:shared/shared_s{s}.zip" for s in SEEDS]
VB_SHARED = [f"vbeam:{p}" for p in SHARED]
VB_B0 = [f"vbeam:depot:b0/b0_s{s}.zip" for s in SEEDS]

summary: dict = {}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    summary[key] = value
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"{key}: {json.dumps(value)}")


def verdict(tag: str, candidates: list[str], baselines: list[str], grid) -> dict | None:
    from locma.harness.ceiling_eval import (  # noqa: PLC0415 -- lazy heavy import
        _disjoint_eval_seeds,
        run_verdict,
    )

    if tag in summary:
        log(f"{tag}: exists, skip")
        return summary[tag]
    n_seeds, gps = grid
    t0 = time.time()
    out = run_verdict(
        candidates,
        baselines,
        seeds=_disjoint_eval_seeds(n_seeds, gps),
        games_per_seed=gps,
        workers=WORKERS,
    )
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    record(tag, out)
    return out


# Retro arms: shelved checkpoints whose critics differ from b0's.
# (vdst-ph/bc excluded: ph's critic is byte-identical to b0, bc's is untrained.)
RETRO = {
    "selfplay_r2": ["vbeam:runs/selfplay-r2.zip"],
    "az_net_0": ["vbeam:runs/az-net-0.zip"],
    "sweep_C": ["vbeam:runs/sweep-C.zip"],
    "vdst_ff": [f"vbeam:runs/vdst-ff_s{s}.zip" for s in SEEDS],
    "cand1": [f"vbeam:depot:cand1/cand1_s{s}.zip" for s in SEEDS],
}


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== zero-training trio driver start ===")

    # Stage 1: retro pilots (cheap; establishes the promotion set early).
    for tag, cands in RETRO.items():
        verdict(f"pilot_{tag}", cands, VB_B0, PILOT)

    # Stage 2: width sweep, cheapest first (w=32 games cost ~4x w=8).
    for w in (4, 16, 32):
        verdict(f"width_{w}", [f"{c},{w}" for c in VB_SHARED], VB_SHARED, FULL)

    # Stage 3: critic ensemble (one candidate arm; ~3x game cost).
    ens = "vbeam:" + "|".join(SHARED)
    verdict("ensemble", [ens], VB_SHARED, FULL)

    # Stage 4: promoted retro arms on the full ruler.
    for tag, cands in RETRO.items():
        pilot = summary.get(f"pilot_{tag}")
        if pilot and pilot["ci_hi"] > 0:
            verdict(f"retro_{tag}", cands, VB_B0, FULL)
        else:
            log(f"retro_{tag}: pilot ci_hi <= 0, not promoted")

    log("=== zero-training trio driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
