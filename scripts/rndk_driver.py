"""E7b dose-response driver: how deep can draft-noise mirror-breaking go?

The E7 discriminator showed mirror-breaking deck diversity improves the
critic as scored under the vbeam planner: rnd4 (13% noise) +0.018, shared
draft +0.026. Known curve points: k=0 (B0, +0.000 by definition) and k=4
(depot:rnd4, +0.018). This driver fills in k=8 (~27%) and k=12 (40%):

  1. train rnd{8,12}_s{0,1,2} = B0 recipe of record + --draft-noise k
     (sequential; ~13.6 min/seed)
  2. paired 40x25 verdicts vs depot:b0 on the STANDARD ruler,
     vbeam (the decision metric) and reactive (regression check --
     at high k the decks degrade, training may start to hurt)

Progress in runs/rndk-overnight.log, results in runs/rndk-summary.json
(rewritten after every step). Idempotent: stages skip when their summary
key exists.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

WORKERS = 19
SEEDS = (0, 1, 2)
KS = (8, 12)
SUMMARY_PATH = "runs/rndk-summary.json"
LOG_PATH = "runs/rndk-overnight.log"

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


def train_seed(k: int, s: int) -> None:
    """B0 recipe of record + --draft-noise k (mirrors the rnd4/shared arms)."""
    from locma.envs.training import train_zoo  # noqa: PLC0415 — lazy heavy import

    out = f"runs/rnd{k}_s{s}.zip"
    if os.path.exists(out) and f"train_k{k}_s{s}" in summary:
        log(f"train k{k} s{s}: exists, skip")
        return
    log(f"train k{k} s{s}: B0 recipe + draft_noise={k} -> {out}")
    t0 = time.time()
    train_zoo(
        steps_per_opponent=200_000,
        out=out,
        seed=s,
        obs_mode="token",
        learning_rate=1e-4,
        target_kl=0.025,
        n_envs=16,
        device="cuda",
        verbose=0,
        draft_noise=k,
    )
    record(f"train_k{k}_s{s}", {"out": out, "minutes": round((time.time() - t0) / 60, 1)})


def verdict(tag: str, candidates: list[str], baselines: list[str]) -> None:
    from locma.harness.ceiling_eval import (  # noqa: PLC0415 — lazy heavy import
        _disjoint_eval_seeds,
        run_verdict,
    )

    if tag in summary:
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    out = run_verdict(
        candidates,
        baselines,
        seeds=_disjoint_eval_seeds(40, 25),
        games_per_seed=25,
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
    log("=== E7b rndK dose-response driver start ===")

    for k in KS:
        for s in SEEDS:
            train_seed(k, s)

    bases = [f"depot:b0/b0_s{s}.zip" for s in SEEDS]
    vb_bases = [f"vbeam:depot:b0/b0_s{s}.zip" for s in SEEDS]
    for k in KS:
        cands = [f"runs/rnd{k}_s{s}.zip" for s in SEEDS]
        # the decision metric first: the critic scored under the planner
        verdict(f"vbeam_k{k}", [f"vbeam:{c}" for c in cands], vb_bases)
        # regression check: does training on degraded decks start to hurt reactively?
        verdict(f"reactive_k{k}", cands, bases)

    log("=== E7b rndK dose-response driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
