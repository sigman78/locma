"""E9 ensemble-distill driver: compress the 3-critic ensemble into one checkpoint.

E8 promoted the 3-critic ensemble (0.926, +0.036/+0.042 confirmed) and pinned
the planner as evaluator-limited. Question: does the ensemble's variance
reduction survive distillation into a single critic (1x evaluator compute)?

Why this distillation differs from every prior null: targets are ensemble
MEAN values -- sibling-differing (unlike E5v2a's constant MC labels), from a
teacher that is not the student's own fixed point (unlike E5v2b), noise-free,
and sampled on exactly the beam's query distribution (root + depth-1 siblings
+ stop-eligible states from vbeam-on-the-ensemble play).

Stages (idempotent):
  1. collect  -- ~100k+ beam-query states, target=ensemble mean, spread=std
  2. train    -- critic-branch-only fine-tune of each shared_sX (policy path
                 byte-identical, so the stop rule and reactive recipe are
                 untouched)
  3. GATE (pre-registered): proceed to verdicts iff pooled val RMSE <
     mean cross-critic spread (else the residual regression noise is as large
     as the variance the ensemble removes -- dead on arrival)
  4. verdicts (40x25 standard ruler):
       vdens vs vbeam:ensemble  -- the decision: how much of +0.042 survives?
       vdens vs vbeam:shared    -- ladder comparability (single vs single)

Progress in runs/ensdist-overnight.log, results in runs/ensdist-summary.json.
Smoke mode: ENSDIST_SMOKE=1 -> tiny grids, separate summary/log files.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

SMOKE = os.environ.get("ENSDIST_SMOKE") == "1"
WORKERS = 19
SEEDS = (0, 1, 2)
SUMMARY_PATH = "runs/ensdist-smoke.json" if SMOKE else "runs/ensdist-summary.json"
LOG_PATH = "runs/ensdist-smoke.log" if SMOKE else "runs/ensdist-overnight.log"
DATA_PATH = "runs/ensdist-smoke-data.npz" if SMOKE else "runs/ensdist-data.npz"
FULL = (2, 2) if SMOKE else (40, 25)
GAMES = 2 if SMOKE else 50  # per opponent, both seats -> 8x game-sides
EPOCHS = 2 if SMOKE else 10

SHARED = [f"depot:shared/shared_s{s}.zip" for s in SEEDS]
VB_SHARED = [f"vbeam:{p}" for p in SHARED]
ENS_SPEC = "vbeam:" + "|".join(SHARED)
DIST = [f"runs/vdens_s{s}.zip" for s in SEEDS]

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


def collect() -> None:
    from locma.envs.vbeam_fvi import collect_ensemble_data  # noqa: PLC0415

    if "collect" in summary and os.path.exists(DATA_PATH):
        log("collect: exists, skip")
        return
    t0 = time.time()
    # Training seeds 30000+: disjoint from eval (1M+/2M+) and from the FVI
    # (10k+) and E4v2 (20k+) collections.
    m = collect_ensemble_data(SHARED, DATA_PATH, games=GAMES, seed=30_000, workers=WORKERS, width=8)
    m["minutes"] = round((time.time() - t0) / 60, 1)
    record("collect", m)


def train() -> None:
    from locma.envs.vbeam_fvi import train_value_head  # noqa: PLC0415

    for s in SEEDS:
        tag = f"train_s{s}"
        if tag in summary and os.path.exists(DIST[s]):
            log(f"{tag}: exists, skip")
            continue
        t0 = time.time()
        metrics = train_value_head(SHARED[s], DATA_PATH, DIST[s], epochs=EPOCHS, seed=s)
        metrics["minutes"] = round((time.time() - t0) / 60, 1)
        record(tag, metrics)


def gate() -> bool:
    """Pre-registered: pooled val RMSE must beat the mean cross-critic spread."""
    if "gate" in summary:
        return summary["gate"]["passed"]
    rmse = math.sqrt(sum(summary[f"train_s{s}"]["val_mse_after"] for s in SEEDS) / len(SEEDS))
    spread = summary["collect"]["mean_spread"]
    passed = bool(rmse < spread)
    record(
        "gate",
        {"pooled_val_rmse": round(rmse, 4), "mean_spread": round(spread, 4), "passed": passed},
    )
    return passed


def verdict(tag: str, candidates: list[str], baselines: list[str]) -> None:
    from locma.harness.ceiling_eval import (  # noqa: PLC0415 -- lazy heavy import
        _disjoint_eval_seeds,
        run_verdict,
    )

    if tag in summary:
        log(f"{tag}: exists, skip")
        return
    n_seeds, gps = FULL
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


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E9 ensemble-distill driver start ===")

    collect()
    train()

    if gate():
        vb_dist = [f"vbeam:{p}" for p in DIST]
        # the decision: recovery fraction of the ensemble's +0.042
        verdict("vdens_vs_ensemble", vb_dist, [ENS_SPEC])
        # ladder comparability: distilled single vs original single critics
        verdict("vdens_vs_shared", vb_dist, VB_SHARED)
    else:
        log("GATE FAILED: distillation residual >= removed variance; verdicts skipped")

    log("=== E9 ensemble-distill driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
