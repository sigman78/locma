"""E6 overnight driver: recurrent PPO (LSTM) vs the reactive B0 ceiling.

The lever: MaskableRecurrentPPO (locma/envs/rppo.py) — LSTM policy/value
trunks that carry hidden state across a game's decisions, so the net can in
principle remember revealed information (opponent's played cards, run-outs)
that the single-frame observation cannot encode. Everything else is the B0
recipe of record: token V0 obs + TokenSetExtractor defaults (dropout 0.1),
lr=1e-4, target_kl=0.025, ent_coef=0.02, n_steps=2048, batch=64, 800k zoo
curriculum, both-seat, n_envs=16, cuda, seeds {0,1,2}.

Stages (idempotent — each skips when its outputs already exist):
  1. train runs/rppo_s{0,1,2}.zip (~112 min/seed at the probed 119 fps)
  2. pilot verdict 10x10  — rppo:runs/rppo_s{s} vs depot:b0/b0_s{s}
  3. full verdict 40x25   — the +-0.03 paired-bootstrap ruler (unconditional:
     E6 wants a real number either way, the LSTM is a foundation lever)

Progress in runs/rppo-overnight.log, machine-readable results in
runs/rppo-summary.json (rewritten after every step so a crash loses nothing).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

WORKERS = 19
SEEDS = (0, 1, 2)
SUMMARY_PATH = "runs/rppo-summary.json"
LOG_PATH = "runs/rppo-overnight.log"

summary: dict = {}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_summary() -> None:
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def record(key: str, value) -> None:
    summary[key] = value
    save_summary()
    log(f"{key}: {json.dumps(value)}")


def b0(s: int) -> str:
    return f"depot:b0/b0_s{s}.zip"


def train_one(seed: int, out: str) -> dict:
    from locma.envs.training import train_zoo  # noqa: PLC0415 — lazy heavy import

    t0 = time.time()
    train_zoo(
        steps_per_opponent=200_000,
        out=out,
        seed=seed,
        obs_mode="token",
        learning_rate=1e-4,
        target_kl=0.025,
        n_envs=16,
        device="cuda",
        verbose=0,
        recurrent=True,
    )
    return {"minutes": round((time.time() - t0) / 60, 1)}


def verdict(tag: str, candidates: list[str], baselines: list[str], seeds: int, games: int) -> dict:
    from locma.harness.ceiling_eval import (  # noqa: PLC0415 — lazy heavy import
        _disjoint_eval_seeds,
        run_verdict,
    )

    t0 = time.time()
    out = run_verdict(
        candidates,
        baselines,
        seeds=_disjoint_eval_seeds(seeds, games),
        games_per_seed=games,
        workers=WORKERS,
    )
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    record(tag, out)
    return out


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E6 rppo overnight driver start ===")

    # ---- Stage 1: train 3 seeds (B0 recipe + LSTM) -----------------------
    for s in SEEDS:
        out = f"runs/rppo_s{s}.zip"
        if os.path.exists(out) and f"train_s{s}" in summary:
            log(f"stage1 s{s}: exists, skip")
            continue
        log(f"stage1 s{s}: train 800k zoo, recurrent, B0 recipe")
        record(f"train_s{s}", train_one(s, out))

    # ---- Stage 2: pilot 10x10 --------------------------------------------
    cands = [f"rppo:runs/rppo_s{s}.zip" for s in SEEDS]
    b0s = [b0(s) for s in SEEDS]
    if "pilot" not in summary:
        log("stage2: pilot verdict 10x10 (rppo vs B0)")
        verdict("pilot", cands, b0s, seeds=10, games=10)

    # ---- Stage 3: full 40x25 ----------------------------------------------
    if "full" not in summary:
        log("stage3: full verdict 40x25 (rppo vs B0)")
        verdict("full", cands, b0s, seeds=40, games=25)

    log("=== E6 rppo overnight driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
