"""E11 zoo-hardening driver: add boardkeep to the training zoo, retrain B0.

E10 found the reactive recipe of record adversarially fragile: the scripted
boardkeep archetype (balanced deck + kill-without-dying trades only) beats
reactive B0 at 0.540 [0.519, 0.562], while the vbeam planner rungs repel it.
The recorded hardening lever: make B0 train against the disciplined trader it
never sees. ZOO_OPPONENTS now ends with "boardkeep" (5 phases, 200k steps
each = 1M total; B0's 4-phase zoo was 800k — the extra phase is the lever, so
the budget delta is part of the treatment, not a confound to remove).

Stages (idempotent -- each skips when its summary key / artifact exists):
  A. train b0k_s{0,1,2}: the B0 recipe of record (token V0, lr=1e-4,
     target_kl=0.025, dropout=0.1 default, n_envs=16, cuda) on the extended zoo.
  B. exploit re-bench, E10 protocol (20x50 seed-mirrored blocks, SEED0=5M --
     same common random numbers as the E10 rows, Wilson 95% CI): all five
     archetypes vs ppo:b0k_s0, plus boardkeep vs b0k_s1/s2 (seed robustness).
  C. standard paired ruler: reactive [b0k_s0..2] vs [depot:b0 s0..2],
     40x25 disjoint eval seeds on a fresh 3M+ range.
  D. planner arm, pilot-gated: vbeam:b0k_sX vs vbeam:depot:b0/b0_sX
     (10x10 pilot; full 40x25 iff pilot mean_delta > -0.10).

Pre-registered gates:
  hardening PASS iff boardkeep-vs-b0k_s0 ci_hi < 0.5 (the exploit no longer
    wins) and no archetype's ci_lo > 0.5 against b0k_s0.
  strength PASS iff stage C mean_delta ci_lo > -0.03 (no meaningful
    avg-hard3 regression).
  b0k replaces the reactive recipe of record iff BOTH pass. The planner arm
  is scored on the usual +0.03 headroom bar but is informational only (the
  deployed planner already repels this family).

Progress in runs/zoohard-overnight.log, results in runs/zoohard-summary.json.
Smoke mode: ZOOHARD_SMOKE=1 -> tiny grids/steps, separate artifact paths.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

SMOKE = os.environ.get("ZOOHARD_SMOKE") == "1"
WORKERS = 19
SEEDS = (0,) if SMOKE else (0, 1, 2)
SUMMARY_PATH = "runs/zoohard-smoke.json" if SMOKE else "runs/zoohard-summary.json"
LOG_PATH = "runs/zoohard-smoke.log" if SMOKE else "runs/zoohard-overnight.log"
MODEL_TMPL = "runs/zoohard-smoke_b0k_s{s}.zip" if SMOKE else "runs/b0k_s{s}.zip"

STEPS_PER_OPP = 2_048 if SMOKE else 200_000
TRAIN_ENVS = 4 if SMOKE else 16
BLOCKS = 2 if SMOKE else 20
GAMES = 2 if SMOKE else 50  # seed-pairs per block; x2 mirrored games
SEED0 = 5_000_000  # matches E10's exploit-bench range (common random numbers)
FULL = (2, 2) if SMOKE else (40, 25)  # (eval seeds, games_per_seed)
PILOT = (2, 2) if SMOKE else (10, 10)
RULER_START = 3_000_000  # fresh eval-seed range (1M standard / 2M E8 confirm used)

EXPLOITS = ("rnddeck", "guardwall", "bufface", "boardkeep", "shell")
B0 = [f"depot:b0/b0_s{s}.zip" for s in (0, 1, 2)]

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


# ---- Stage A: training ------------------------------------------------------


def train_seed(s: int) -> None:
    """B0 recipe of record (see depot:b0 provenance) on the boardkeep-extended zoo."""
    from locma.envs.training import ZOO_OPPONENTS, train_zoo  # noqa: PLC0415

    out = model_path(s)
    if os.path.exists(out) and f"train_s{s}" in summary:
        log(f"train s{s}: exists, skip")
        return
    log(f"train s{s}: B0 recipe on zoo {ZOO_OPPONENTS} -> {out}")
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
    )
    record(
        f"train_s{s}",
        {
            "out": out,
            "zoo": list(ZOO_OPPONENTS),
            "steps": STEPS_PER_OPP * len(ZOO_OPPONENTS),
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


# ---- Stage B: exploit re-bench (E10 protocol) -------------------------------

_CACHE: dict = {}


def _cell(exploit: str, defender: str, seed: int, games: int) -> tuple[int, int]:
    """Picklable pool unit: one seed block of run_match(exploit, defender)."""
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    for spec in (exploit, defender):
        if spec not in _CACHE:
            _CACHE[spec] = make_policy(spec)
    res = run_match(_CACHE[exploit], _CACHE[defender], games=games, seed=seed)
    return res.wins_a, res.games


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def bench_pair(ex, tag: str, exploit: str, defender: str) -> None:
    if tag in summary:
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    seeds = [SEED0 + b * GAMES for b in range(BLOCKS)]
    results = list(ex.map(_cell, [exploit] * BLOCKS, [defender] * BLOCKS, seeds, [GAMES] * BLOCKS))
    wins = sum(w for w, _ in results)
    n = sum(g for _, g in results)
    wr, lo, hi = _wilson(wins, n)
    record(
        tag,
        {
            "wr": round(wr, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "games": n,
            "exploit_wins": bool(lo > 0.5),
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


# ---- Stages C/D: paired ruler verdicts --------------------------------------


def verdict(tag: str, candidates: list[str], baselines: list[str], grid, start: int) -> dict | None:
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


# ---- Pre-registered gate evaluation -----------------------------------------


def gates() -> None:
    if "e11_gates" in summary:
        log("e11_gates: exists, skip")
        return
    bk = summary[f"boardkeep__vs__ppo:{model_path(0)}"]
    hardening = bk["ci_hi"] < 0.5 and not any(
        summary[f"{e}__vs__ppo:{model_path(0)}"]["exploit_wins"] for e in EXPLOITS
    )
    ruler = summary["full_reactive_standard"]
    strength = ruler["ci_lo"] > -0.03
    record(
        "e11_gates",
        {
            "hardening_pass": bool(hardening),
            "boardkeep_wr": bk["wr"],
            "boardkeep_ci_hi": bk["ci_hi"],
            "strength_pass": bool(strength),
            "ruler_mean_delta": ruler["mean_delta"],
            "ruler_ci_lo": ruler["ci_lo"],
            "promote_reactive_recipe": bool(hardening and strength),
        },
    )


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E11 zoo-hardening driver start ===")

    # Stage A: train the hardened seeds (sequential; cuda + 16 envs each).
    for s in SEEDS:
        train_seed(s)

    # Stage B: exploit re-bench vs the hardened net.
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_eval_worker) as ex:
        for exploit in EXPLOITS:
            bench_pair(ex, f"{exploit}__vs__ppo:{model_path(0)}", exploit, f"ppo:{model_path(0)}")
        for s in SEEDS[1:]:  # the headline exploit vs the other seeds
            bench_pair(
                ex, f"boardkeep__vs__ppo:{model_path(s)}", "boardkeep", f"ppo:{model_path(s)}"
            )

    # Stage C: standard paired ruler, fresh 3M+ eval-seed range.
    cands = [model_path(s) for s in SEEDS]
    bases = B0[: len(SEEDS)]
    verdict("full_reactive_standard", cands, bases, FULL, start=RULER_START)

    # Stage D: planner arm, pilot-gated (does boardkeep data help the critic?).
    vb_cands = [f"vbeam:{model_path(s)}" for s in SEEDS]
    vb_bases = [f"vbeam:{b}" for b in bases]
    pilot = verdict("pilot_vbeam_standard", vb_cands, vb_bases, PILOT, start=RULER_START)
    if pilot["mean_delta"] > -0.10:
        verdict("full_vbeam_standard", vb_cands, vb_bases, FULL, start=RULER_START)
    else:
        record("full_vbeam_standard_skipped", "pilot clearly negative (< -0.10)")

    gates()
    log("=== E11 zoo-hardening driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
