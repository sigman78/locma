"""E13 stacking driver: boardkeep zoo + shared draft in one training run (bks).

E11/E12 established two independent training-side critic levers reaching
~the same place: shared-draft data (vbeam:shared 0.890, E7) and the
boardkeep-extended zoo (vbeam:b0k 0.886, default draft). Their mechanisms
differ (deck asymmetry vs opponent discipline), so unlike E7c's
shared+rnd4 non-stack (same mechanism twice) this combination is not ruled
out. E12's mixed-ensemble null (b0k substitutes 1:1 for shared) hints their
critics' ERRORS overlap -- ensemble-level evidence; this is the
training-level test.

bks = the B0 recipe of record (token V0, lr=1e-4, target_kl=0.025, cuda,
n_envs=16) on the boardkeep-extended ZOO_OPPONENTS with shared_draft=True,
3 seeds x 1M steps.

Stages (idempotent -- each skips when its summary key / artifact exists):
  A. train bks_s{0,1,2}.
  B. PRIMARY (the stacking question): vbeam:bks_sX vs vbeam:depot:shared_sX,
     10x10 pilot (ci_hi > 0 gates the full), full 40x25 at 7M+ seeds.
  C. reactive bks_sX vs depot:b0k (the new reactive RoR), full 40x25 at 7M+
     -- informational (E7 showed shared training's reactive transfer ~null).
  D. conditional: iff stage B full ci_lo > 0, ens(bks x3) vs the ensemble
     RoR (vbeam:shared_s0|s1|s2), pilot then full at 7M+.
  E. fresh-anchor confirms at 8M+ for any CI-positive full in B or D.

Pre-registered gates:
  STACK iff stage B full AND confirm both ci_lo > 0 (the levers compose).
  ens(bks x3) promotable over the RoR (compute parity) iff its full AND
    confirm both ci_lo > 0.
  Stage C is recorded, not gated (no reactive promotion claim from E13).

Seed ranges already used: 1M standard, 2M E8, 3M E11, 4M/6M E12, 5M exploit
match seeds -- E13 uses 7M primary / 8M confirm.

Progress in runs/bks-overnight.log, results in runs/bks-summary.json.
Smoke mode: BKS_SMOKE=1 -> tiny grids/steps, separate artifact paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("BKS_SMOKE") == "1"
WORKERS = 19
SEEDS = (0,) if SMOKE else (0, 1, 2)
SUMMARY_PATH = "runs/bks-smoke.json" if SMOKE else "runs/bks-summary.json"
LOG_PATH = "runs/bks-smoke.log" if SMOKE else "runs/bks-overnight.log"
MODEL_TMPL = "runs/bks-smoke_s{s}.zip" if SMOKE else "runs/bks_s{s}.zip"

STEPS_PER_OPP = 2_048 if SMOKE else 200_000
TRAIN_ENVS = 4 if SMOKE else 16
FULL = (2, 2) if SMOKE else (40, 25)  # (eval seeds, games_per_seed)
PILOT = (2, 2) if SMOKE else (10, 10)
PRIMARY_START = 7_000_000
CONFIRM_START = 8_000_000

ALL_SEEDS = (0, 1, 2)
SHARED = [f"depot:shared/shared_s{s}.zip" for s in ALL_SEEDS]
B0K = [f"depot:b0k/b0k_s{s}.zip" for s in ALL_SEEDS]
ENS_ROR = "vbeam:" + "|".join(SHARED)  # the 0.926 planner recipe of record

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


def train_seed(s: int) -> None:
    """B0 recipe on the boardkeep-extended zoo, shared draft variant."""
    from locma.envs.training import ZOO_OPPONENTS, train_zoo  # noqa: PLC0415

    out = model_path(s)
    if os.path.exists(out) and f"train_s{s}" in summary:
        log(f"train s{s}: exists, skip")
        return
    log(f"train s{s}: B0 recipe + shared draft on zoo {ZOO_OPPONENTS} -> {out}")
    t0 = time.time()
    train_zoo(
        steps_per_opponent=STEPS_PER_OPP,
        out=out,
        seed=s,
        obs_mode="token",
        learning_rate=1e-4,
        target_kl=0.025,
        n_envs=TRAIN_ENVS,
        device="cuda",
        verbose=0,
        shared_draft=True,
    )
    record(
        f"train_s{s}",
        {
            "out": out,
            "zoo": list(ZOO_OPPONENTS),
            "shared_draft": True,
            "steps": STEPS_PER_OPP * len(ZOO_OPPONENTS),
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


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


def gates() -> None:
    if "e13_gates" in summary:
        log("e13_gates: exists, skip")
        return
    g: dict = {}
    full = summary.get("full_vbeam_bks")
    conf = summary.get("confirm_vbeam_bks_8M")
    if full is None:
        g["stack"] = "pilot-gated out"
    elif full["ci_lo"] <= 0:
        g["stack"] = f"NO STACK: full not CI-positive ({full['mean_delta']})"
    elif conf is None:
        g["stack"] = "full CI-positive but unconfirmed"
    else:
        ok = conf["ci_lo"] > 0
        g["stack"] = f"{'STACK CONFIRMED' if ok else 'confirm failed'} ({conf['mean_delta']})"
    ens_full = summary.get("full_ens_bks3")
    ens_conf = summary.get("confirm_ens_bks3_8M")
    if ens_full is None:
        g["ens_bks3"] = "not run (stage B not CI-positive) or pilot-gated out"
    elif ens_full["ci_lo"] <= 0:
        g["ens_bks3"] = f"full not CI-positive ({ens_full['mean_delta']})"
    elif ens_conf is None:
        g["ens_bks3"] = "full CI-positive but unconfirmed"
    else:
        ok = ens_conf["ci_lo"] > 0
        g["ens_bks3"] = (
            f"{'PROMOTE over RoR' if ok else 'confirm failed'} ({ens_conf['mean_delta']})"
        )
    reac = summary.get("full_reactive_bks_vs_b0k")
    if reac is not None:
        g["reactive_vs_b0k"] = reac["mean_delta"]
    record("e13_gates", g)


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E13 bks stacking driver start ===")

    # Stage A: train.
    for s in SEEDS:
        train_seed(s)

    cands = [model_path(s) for s in SEEDS]
    vb_cands = [f"vbeam:{c}" for c in cands]
    vb_shared = [f"vbeam:{p}" for p in SHARED[: len(SEEDS)]]

    # Stage B: the stacking question, pilot-gated.
    pilot = verdict("pilot_vbeam_bks", vb_cands, vb_shared, PILOT, start=PRIMARY_START)
    if pilot["ci_hi"] > 0:
        verdict("full_vbeam_bks", vb_cands, vb_shared, FULL, start=PRIMARY_START)
    else:
        log("full_vbeam_bks: pilot ci_hi <= 0, not promoted")

    # Stage C: reactive side vs the new reactive RoR (informational).
    verdict("full_reactive_bks_vs_b0k", cands, B0K[: len(SEEDS)], FULL, start=PRIMARY_START)

    # Stage D: conditional ensemble arm.
    full = summary.get("full_vbeam_bks")
    if full and full["ci_lo"] > 0:
        ens_bks = "vbeam:" + "|".join(cands)
        p = verdict("pilot_ens_bks3", [ens_bks], [ENS_ROR], PILOT, start=PRIMARY_START)
        if p["ci_hi"] > 0:
            verdict("full_ens_bks3", [ens_bks], [ENS_ROR], FULL, start=PRIMARY_START)
    else:
        log("ens_bks3: stage B not CI-positive, skipped")

    # Stage E: fresh-anchor confirms for CI-positive fulls.
    if summary.get("full_vbeam_bks", {}).get("ci_lo", 0) > 0:
        verdict("confirm_vbeam_bks_8M", vb_cands, vb_shared, FULL, start=CONFIRM_START)
    if summary.get("full_ens_bks3", {}).get("ci_lo", 0) > 0:
        ens_bks = "vbeam:" + "|".join(cands)
        verdict("confirm_ens_bks3_8M", [ens_bks], [ENS_ROR], FULL, start=CONFIRM_START)

    gates()
    log("=== E13 bks stacking driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
