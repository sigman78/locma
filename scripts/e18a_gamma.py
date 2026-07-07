"""E18a gamma=1.0 retrain: the cheapest transferable ByteRL ablation.

ByteRL (arXiv 2303.04096 / 2303.05197) reports +7% from switching the discount
to gamma=1.0 -- in a game where the ONLY reward is the terminal win/loss, any
gamma < 1 silently down-weights that signal at early decisions. Our games run
~30-60 agent steps, so at the recipe's gamma=0.99 the terminal reward reaches
turn-1 decisions scaled by ~0.55-0.74. Undiscounting is a single-factor change
to the reactive recipe of record and compounds if it works: a stronger reactive
net is also a stronger vbeam critic (the value head at gamma=1 estimates win
probability directly, which is exactly what the beam ranks with).

Treatment: the b0k recipe of record (E11/E12 provenance: token V0 obs,
lr=1e-4, target_kl=0.025, n_envs=16, cuda, 5-phase zoo ending in boardkeep,
200k steps/phase = 1M total) with gamma=1.0 instead of the SB3-default 0.99.
Seeds 0/1/2, trained sequentially (one cuda trainer at a time).

Stages (idempotent, resumable via runs/e18a-summary.json):
  A. train g1_s{0,1,2} -> runs/e18a_g1_s{s}.zip (~34 min/seed at E11 rates).
  B. reactive ruler: [g1_s0..2] vs [depot:b0k s0..2], 40x25 @ 16M anchors.
  C. planner arm, pilot-gated: [vbeam:g1_sX] vs [vbeam:depot:b0k_sX]
     (10x10 @ 16M; full 40x25 iff pilot mean_delta > 0) -- the critic read.
  D. confirm on fresh 17M anchors iff stage B ci_lo > 0.

Pre-registered gates:
  reactive promotion candidate iff B ci_lo > 0 AND D ci_lo > 0 (E12 pattern).
  critic read is informational; a CI-positive full planner arm flags g1 as
  ensemble-member material (E12: ensembles saturate at 3, so any swap-in has
  to beat a shared critic head-to-head first -- not tested here).

Seed ranges: 16M primary / 17M confirm (1M-15M all used: standard/E8/E11/
exploit/E14a/E15/E16a/E17). Smoke: E18A_SMOKE=1 -> tiny steps/grids,
separate artifact paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E18A_SMOKE") == "1"
WORKERS = 19
SEEDS = (0,) if SMOKE else (0, 1, 2)
SUMMARY_PATH = "runs/e18a-smoke.json" if SMOKE else "runs/e18a-summary.json"
LOG_PATH = "runs/e18a-smoke.log" if SMOKE else "runs/e18a.log"
MODEL_TMPL = "runs/e18a-smoke_g1_s{s}.zip" if SMOKE else "runs/e18a_g1_s{s}.zip"

STEPS_PER_OPP = 2_048 if SMOKE else 200_000
TRAIN_ENVS = 4 if SMOKE else 16
FULL = (2, 2) if SMOKE else (40, 25)  # (eval seeds, games_per_seed)
PILOT = (2, 2) if SMOKE else (10, 10)
PRIMARY_START = 16_000_000
CONFIRM_START = 17_000_000

B0K = [f"depot:b0k/b0k_s{s}.zip" for s in (0, 1, 2)]

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


def model_path(s: int) -> str:
    return MODEL_TMPL.format(s=s)


# ---- Stage A: training -------------------------------------------------------


def train_seed(s: int) -> None:
    """b0k recipe of record with the single-factor gamma=1.0 change."""
    from locma.envs.training import ZOO_OPPONENTS, train_zoo  # noqa: PLC0415

    out = model_path(s)
    if os.path.exists(out) and f"train_s{s}" in summary:
        log(f"train s{s}: exists, skip")
        return
    log(f"train s{s}: b0k recipe + gamma=1.0 on zoo {ZOO_OPPONENTS} -> {out}")
    t0 = time.time()
    train_zoo(
        steps_per_opponent=STEPS_PER_OPP,
        out=out,
        seed=s,
        obs_mode="token",
        learning_rate=1e-4,
        target_kl=0.025,
        gamma=1.0,  # the treatment; everything else matches depot:b0k provenance
        n_envs=TRAIN_ENVS,
        device="cuda",
        verbose=0,
    )
    record(
        f"train_s{s}",
        {
            "out": out,
            "gamma": 1.0,
            "zoo": list(ZOO_OPPONENTS),
            "steps": STEPS_PER_OPP * len(ZOO_OPPONENTS),
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


# ---- Verdict wrapper (E12 pattern) -------------------------------------------


def verdict(tag: str, candidates: list[str], baselines: list[str], grid, start: int) -> dict:
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
        seeds=_disjoint_eval_seeds(n_seeds, gps, start=start),
        games_per_seed=gps,
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
    log("=== E18a gamma=1.0 retrain start ===")

    # Stage A: train the undiscounted seeds (sequential; cuda + 16 envs each).
    for s in SEEDS:
        train_seed(s)

    cands = [model_path(s) for s in SEEDS]
    bases = B0K[: len(SEEDS)]

    # Stage B: reactive ruler vs the recipe of record.
    full = verdict("full_reactive", cands, bases, FULL, start=PRIMARY_START)

    # Stage C: planner arm (critic read), pilot-gated.
    vb_cands = [f"vbeam:{m}" for m in cands]
    vb_bases = [f"vbeam:{b}" for b in bases]
    pilot = verdict("pilot_vbeam", vb_cands, vb_bases, PILOT, start=PRIMARY_START)
    if pilot["mean_delta"] > 0:
        verdict("full_vbeam", vb_cands, vb_bases, FULL, start=PRIMARY_START)
    else:
        record("full_vbeam_skipped", "pilot not positive")

    # Stage D: fresh-anchor confirm iff the primary ruler is CI-positive.
    if full["ci_lo"] > 0:
        verdict("confirm_reactive_17M", cands, bases, FULL, start=CONFIRM_START)

    # Pre-registered gates.
    g: dict = {
        "reactive_delta": full["mean_delta"],
        "reactive_ci": [full["ci_lo"], full["ci_hi"]],
        "primary_ci_positive": bool(full["ci_lo"] > 0),
    }
    conf = summary.get("confirm_reactive_17M")
    g["promote_candidate"] = bool(full["ci_lo"] > 0 and conf is not None and conf["ci_lo"] > 0)
    if conf is not None:
        g["confirm_delta"] = conf["mean_delta"]
        g["confirm_ci"] = [conf["ci_lo"], conf["ci_hi"]]
    fv = summary.get("full_vbeam")
    g["critic_flag"] = bool(fv is not None and fv["ci_lo"] > 0)
    record("e18a_gates", g)

    log("=== E18a gamma=1.0 retrain DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
