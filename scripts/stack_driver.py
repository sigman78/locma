"""E7c stacking driver: does draft-noise variance stack on shared-draft structure?

E7 promoted shared (+0.026 under vbeam); E7b showed plain noise saturates at
~+0.018-0.023 regardless of dose. The one untested configuration that could
beat +0.026 is BOTH at once: contested offers (structure) + rnd4 (variance).
A null here pins the whole effect on mirror-breaking; a gain means the two
mechanisms are additive.

Stages (idempotent):
  1. train shrnd4_s{0,1,2} = B0 recipe + shared_draft=True + draft_noise=4
  2. verdicts (40x25, standard ruler):
       vbeam:shrnd4 vs vbeam:depot:b0      -- comparability with all E7/E7b deltas
       vbeam:shrnd4 vs vbeam:depot:shared  -- the decision: does stacking beat shared?
       shrnd4 vs depot:b0 (reactive)       -- regression check

Progress in runs/stack-overnight.log, results in runs/stack-summary.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

WORKERS = 19
SEEDS = (0, 1, 2)
SUMMARY_PATH = "runs/stack-summary.json"
LOG_PATH = "runs/stack-overnight.log"

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


def train_seed(s: int) -> None:
    """B0 recipe of record + shared draft + 4 noisy picks per deck."""
    from locma.envs.training import train_zoo  # noqa: PLC0415 — lazy heavy import

    out = f"runs/shrnd4_s{s}.zip"
    if os.path.exists(out) and f"train_s{s}" in summary:
        log(f"train s{s}: exists, skip")
        return
    log(f"train s{s}: B0 recipe + shared_draft + draft_noise=4 -> {out}")
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
        shared_draft=True,
        draft_noise=4,
    )
    record(f"train_s{s}", {"out": out, "minutes": round((time.time() - t0) / 60, 1)})


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
    log("=== E7c shared+rnd4 stacking driver start ===")

    for s in SEEDS:
        train_seed(s)

    cands = [f"runs/shrnd4_s{s}.zip" for s in SEEDS]
    vb_cands = [f"vbeam:{c}" for c in cands]
    vb_b0 = [f"vbeam:depot:b0/b0_s{s}.zip" for s in SEEDS]
    vb_shared = [f"vbeam:depot:shared/shared_s{s}.zip" for s in SEEDS]

    # comparable to every E7/E7b delta (same baseline, same ruler)
    verdict("vbeam_stack_vs_b0", vb_cands, vb_b0)
    # the decision metric: direct paired comparison against the recipe of record
    verdict("vbeam_stack_vs_shared", vb_cands, vb_shared)
    # regression check (shared alone was +0.003 null; a negative here = noise tax)
    verdict("reactive_stack_vs_b0", cands, [f"depot:b0/b0_s{s}.zip" for s in SEEDS])

    log("=== E7c shared+rnd4 stacking driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
