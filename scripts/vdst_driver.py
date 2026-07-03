"""E4v2 overnight driver: distill the vbeam teacher + expert iteration.

Stages (idempotent — each skips when its outputs already exist):
  1. collect vbeam(b0_s{s}) play data, zoo + self-play, s in {0,1,2}
  2. train PH (policy-head), BC (scratch), FF (warm full FT) students
  3. pilot evals 10x10 (reactive PH/BC/FF; planner vbeam:PH)
  4. full 40x25 verdicts (reactive always; planner if pilot > -0.02)
  5. conditional round 2 (EXIT): re-collect from vbeam:PH, train ph2, eval

All artifacts land in runs/ (vdst-*), progress in runs/vdst-overnight.log,
machine-readable results in runs/vdst-summary.json (rewritten after every
step so a crash loses nothing).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

WORKERS = 19
SEEDS = (0, 1, 2)
ZOO = ("greedy", "scripted", "max-guard", "max-attack")
SUMMARY_PATH = "runs/vdst-summary.json"
LOG_PATH = "runs/vdst-overnight.log"

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


def collect(teacher_model: str, out: str, seed: int) -> dict:
    """Zoo + self-play collection for one teacher model."""
    from locma.envs.vbeam_fvi import collect_value_data  # noqa: PLC0415 — lazy heavy import

    t0 = time.time()
    spec = f"vbeam:{teacher_model}"
    manifest = collect_value_data(
        spec,
        out,
        opponents=(*ZOO, spec),
        games=400,
        seed=seed,
        workers=WORKERS,
    )
    return {
        "n_examples": manifest["n_examples"],
        "failed_games": manifest["failed_games"],
        "minutes": round((time.time() - t0) / 60, 1),
    }


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
    log("=== E4v2 overnight driver start ===")

    # ---- Stage 1: collection --------------------------------------------
    for s in SEEDS:
        out = f"runs/vdst-data-s{s}.npz"
        if os.path.exists(out) and f"collect_s{s}" in summary:
            log(f"stage1 s{s}: exists, skip")
            continue
        log(f"stage1 s{s}: collecting from vbeam:{b0(s)} (zoo + self-play, 400/opp)")
        record(f"collect_s{s}", collect(b0(s), out, seed=20000 + 500 * s))

    # ---- Stage 2: training ----------------------------------------------
    from locma.envs.distill import behavior_clone  # noqa: PLC0415 — lazy heavy import
    from locma.envs.vbeam_distill import train_policy_head  # noqa: PLC0415

    for s in SEEDS:
        data = f"runs/vdst-data-s{s}.npz"

        ph = f"runs/vdst-ph_s{s}.zip"
        if not (os.path.exists(ph) and f"train_ph_s{s}" in summary):
            log(f"stage2 s{s}: PH policy-head fine-tune")
            m = train_policy_head(b0(s), data, ph, epochs=10, verbose=1)
            record(
                f"train_ph_s{s}",
                {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            )

        bc = f"runs/vdst-bc_s{s}.zip"
        if not (os.path.exists(bc) and f"train_bc_s{s}" in summary):
            log(f"stage2 s{s}: BC from scratch")
            m = behavior_clone(data=data, out=bc, epochs=10, batch=256, seed=s, verbose=0)
            record(
                f"train_bc_s{s}",
                {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            )

        ff = f"runs/vdst-ff_s{s}.zip"
        if not (os.path.exists(ff) and f"train_ff_s{s}" in summary):
            log(f"stage2 s{s}: FF warm-start full fine-tune (lr 1e-4)")
            m = behavior_clone(
                data=data,
                out=ff,
                epochs=10,
                batch=256,
                lr=1e-4,
                seed=s,
                verbose=0,
                init_model=b0(s),
            )
            record(
                f"train_ff_s{s}",
                {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            )

    # ---- Stage 3: pilots (10x10) ----------------------------------------
    b0s = [b0(s) for s in SEEDS]
    arms = {
        "ph": [f"runs/vdst-ph_s{s}.zip" for s in SEEDS],
        "bc": [f"runs/vdst-bc_s{s}.zip" for s in SEEDS],
        "ff": [f"runs/vdst-ff_s{s}.zip" for s in SEEDS],
    }
    for arm, cands in arms.items():
        tag = f"pilot_reactive_{arm}"
        if tag not in summary:
            log(f"stage3: pilot reactive {arm}")
            verdict(tag, cands, b0s, seeds=10, games=10)

    vb_cands = [f"vbeam:runs/vdst-ph_s{s}.zip" for s in SEEDS]
    vb_base = [f"vbeam:{b0(s)}" for s in SEEDS]
    if "pilot_vbeam_ph" not in summary:
        log("stage3: pilot planner vbeam:PH vs vbeam:B0 (EXIT)")
        verdict("pilot_vbeam_ph", vb_cands, vb_base, seeds=10, games=10)

    # ---- Stage 4: full verdicts (40x25) ----------------------------------
    for arm, cands in arms.items():
        tag = f"full_reactive_{arm}"
        if tag not in summary:
            log(f"stage4: full reactive {arm}")
            verdict(tag, cands, b0s, seeds=40, games=25)

    if summary["pilot_vbeam_ph"]["mean_delta"] > -0.02:
        if "full_vbeam_ph" not in summary:
            log("stage4: full planner vbeam:PH")
            verdict("full_vbeam_ph", vb_cands, vb_base, seeds=40, games=25)
    else:
        record("full_vbeam_ph_skipped", "pilot clearly negative")

    # ---- Stage 5: conditional round 2 (EXIT compounding) -----------------
    exit_signal = summary["pilot_vbeam_ph"]["mean_delta"] >= 0.01
    reactive_signal = max(summary[f"full_reactive_{a}"]["mean_delta"] for a in arms) >= 0.03
    record("round2_trigger", {"exit_signal": exit_signal, "reactive_signal": reactive_signal})

    if exit_signal or reactive_signal:
        for s in SEEDS:
            out = f"runs/vdst2-data-s{s}.npz"
            if not (os.path.exists(out) and f"collect2_s{s}" in summary):
                log(f"stage5 s{s}: round-2 collection from vbeam:runs/vdst-ph_s{s}.zip")
                record(
                    f"collect2_s{s}", collect(f"runs/vdst-ph_s{s}.zip", out, seed=30000 + 500 * s)
                )
            ph2 = f"runs/vdst-ph2_s{s}.zip"
            if not (os.path.exists(ph2) and f"train_ph2_s{s}" in summary):
                m = train_policy_head(f"runs/vdst-ph_s{s}.zip", out, ph2, epochs=10, verbose=1)
                record(
                    f"train_ph2_s{s}",
                    {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                )

        vb2 = [f"vbeam:runs/vdst-ph2_s{s}.zip" for s in SEEDS]
        if "pilot_vbeam_ph2" not in summary:
            verdict("pilot_vbeam_ph2", vb2, vb_base, seeds=10, games=10)
        if summary["pilot_vbeam_ph2"]["mean_delta"] > -0.02 and "full_vbeam_ph2" not in summary:
            verdict("full_vbeam_ph2", vb2, vb_base, seeds=40, games=25)
        ph2s = [f"runs/vdst-ph2_s{s}.zip" for s in SEEDS]
        if "full_reactive_ph2" not in summary:
            verdict("full_reactive_ph2", ph2s, b0s, seeds=40, games=25)
    else:
        record("round2_skipped", "no EXIT or reactive signal")

    log("=== E4v2 overnight driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
