"""E21 pilot: swap the zoo's hardest phase from a scripted archetype to an
"advanced player" -- the frozen b0k battle net paired with E20's zero-
inference learned-draft heuristic (depot:edraft) -- and see whether a NEW PPO
generation trained against that sparring partner beats depot:b0k itself.

Motivation. Every zoo opponent to date (greedy/scripted/max-guard/max-attack/
boardkeep, ZOO_OPPONENTS in locma/envs/training.py) is a scripted heuristic.
E11 already showed one such addition (boardkeep) buys parity against its own
exploit, never refutation ("more gradient vs a fixed exploit buys parity, not
capability" -- worklog E11). The untried lever: replace the curriculum's last
phase with an opponent that is not a hand-written archetype but the strongest
CHEAP net-based play available -- an old battle-net generation (depot:b0k)
equipped with a better draft (depot:edraft, E20 -- zero net inference at
draft time, so this costs nothing extra over a normal b0k opponent). This is
a light form of generational self-play: train gen N+1 against gen N wearing a
better deck.

Treatment (isolates opponent IDENTITY, not budget): ZOO_OPPONENTS[:-1] +
(ADV_sX,) where ADV_sX = "ppo:depot:b0k/b0k_sX.zip,depot:edraft/
e20-elicit-fit.json" -- same 5 phases, same 200k/phase, same 1M total budget
as depot:b0k's own recipe (token V0, lr=1e-4, target_kl=0.025, n_envs=16,
cuda). Only the 5th phase's opponent spec differs from b0k's own recipe
(which trained phase 5 against "boardkeep"). Models: runs/e21_adv_s{0,1,2}.zip.

Stages (idempotent, resumable via runs/e21-summary.json):
  A. train e21_adv_s{0,1,2} (sequential, ~35-50 min/seed, matches b0k budget).
  B. reactive pilot ruler (10x10 @ 24M anchors): ppo:e21_adv_sX (balanced
     draft, default) vs ppo:depot:b0k/b0k_sX (balanced draft, default) -- the
     apples-to-apples footing b0k's own 0.683 number was measured on.
  C. planner/critic pilot (10x10 @ 25M anchors): vbeam:e21_adv_sX,8,20 vs
     vbeam:depot:b0k/b0k_sX,8,20 -- does a stronger sparring partner produce
     a better critic (the E7/E19 pattern: value heads are data-sensitive)?
  D. boardkeep guard re-read (E10/E11/E18c/E19 protocol: 20x50 mirrored
     blocks, SEED0=5_000_000 common random numbers vs prior rows). e21_adv
     never trains against boardkeep directly (it was swapped OUT) -- this
     checks whether the advanced opponent's generality substitutes for
     boardkeep-specific hardening, regresses toward E10's original 0.540, or
     lands somewhere between.

This is a PILOT ONLY: no full 40x25 grid, no fresh-anchor confirm, no
promotion gate. It answers "is there a signal worth a full run", nothing
stronger. If stage B or C is CI-positive, the natural follow-up is the full
E19-style gate ladder (full ruler + confirm) before any promotion talk.

Seed ranges: eval anchors 24M (reactive) / 25M (planner) -- fresh, after
E19's 20M/21M and E20's 22M/23M. Smoke: E21_SMOKE=1 -> tiny steps/grids,
separate artifact paths.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E21_SMOKE") == "1"
WORKERS = 19
SEEDS = (0,) if SMOKE else (0, 1, 2)
SUMMARY_PATH = "runs/e21-smoke.json" if SMOKE else "runs/e21-summary.json"
LOG_PATH = "runs/e21-smoke.log" if SMOKE else "runs/e21.log"
MODEL_TMPL = "runs/e21-smoke_adv_s{s}.zip" if SMOKE else "runs/e21_adv_s{s}.zip"

STEPS_PER_OPP = 2_048 if SMOKE else 200_000
TRAIN_ENVS = 4 if SMOKE else 16
PILOT = (2, 2) if SMOKE else (10, 10)
REACTIVE_START = 24_000_000  # fresh range, after E20's 22M/23M
PLANNER_START = 25_000_000
GUARD_BLOCKS = 2 if SMOKE else 20
GUARD_GAMES = 2 if SMOKE else 50
GUARD_SEED0 = 5_000_000  # E10/E11/E18c/E19 common random numbers

B0K = [f"depot:b0k/b0k_s{s}.zip" for s in (0, 1, 2)]
EDRAFT = "depot:edraft/e20-elicit-fit.json"
# The "advanced player": old battle-net generation + E20's zero-inference
# learned-draft heuristic. Matched by seed index (trainee s faces b0k_s /
# same-lineage opponent, wearing the better draft).
ADV = [f"ppo:{B0K[s]},{EDRAFT}" for s in (0, 1, 2)]

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
    """b0k recipe of record, with the zoo's last phase swapped to the advanced opponent."""
    from locma.envs.training import ZOO_OPPONENTS, train_zoo  # noqa: PLC0415

    out = model_path(s)
    if os.path.exists(out) and f"train_s{s}" in summary:
        log(f"train s{s}: exists, skip")
        return
    zoo = (*ZOO_OPPONENTS[:-1], ADV[s])
    log(f"train s{s}: b0k recipe, zoo {zoo} -> {out}")
    t0 = time.time()
    train_zoo(
        opponents=zoo,
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
            "zoo": list(zoo),
            "steps": STEPS_PER_OPP * len(zoo),
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


# ---- Verdict wrapper (E11/E19 pattern) ---------------------------------------


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


# ---- Stage D: boardkeep guard (E10/E11/E18c/E19 protocol) -------------------

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


def boardkeep_guard() -> None:
    tag = "guard_boardkeep"
    if tag in summary:
        log(f"{tag}: exists, skip")
        return
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

    defender = f"ppo:{model_path(0)}"
    t0 = time.time()
    seeds = [GUARD_SEED0 + b * GUARD_GAMES for b in range(GUARD_BLOCKS)]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_eval_worker) as ex:
        results = list(
            ex.map(
                _cell,
                ["boardkeep"] * GUARD_BLOCKS,
                [defender] * GUARD_BLOCKS,
                seeds,
                [GUARD_GAMES] * GUARD_BLOCKS,
            )
        )
    wins = sum(w for w, _ in results)
    n = sum(g for _, g in results)
    wr, lo, hi = _wilson(wins, n)
    record(
        tag,
        {
            "defender": defender,
            "wr": round(wr, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "games": n,
            "e10_b0_wr": 0.540,  # original reactive B0, same protocol
            "e11_b0k_wr": 0.512,  # boardkeep explicitly in the zoo
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E21 advanced-opponent zoo pilot start ===")

    # Stage A: train (sequential; each run owns the GPU + 16 subprocess envs).
    for s in SEEDS:
        train_seed(s)

    cands = [f"ppo:{model_path(s)}" for s in SEEDS]
    bases = [f"ppo:{B0K[s]}" for s in SEEDS]

    # Stage B: reactive pilot ruler, both sides on the default balanced draft
    # (the footing depot:b0k's own 0.683 number was measured on).
    reactive = verdict("pilot_reactive", cands, bases, PILOT, start=REACTIVE_START)

    # Stage C: planner/critic pilot, single critic per side.
    vb_cands = [f"vbeam:{model_path(s)},8,20" for s in SEEDS]
    vb_bases = [f"vbeam:{b},8,20" for b in B0K[: len(SEEDS)]]
    planner = verdict("pilot_planner", vb_cands, vb_bases, PILOT, start=PLANNER_START)

    # Stage D: boardkeep guard -- does the advanced opponent's generality
    # substitute for training against boardkeep directly (which this zoo no
    # longer does)?
    boardkeep_guard()

    guard = summary["guard_boardkeep"]
    record(
        "e21_read",
        {
            "reactive_pilot_delta": reactive["mean_delta"],
            "reactive_pilot_ci": [reactive["ci_lo"], reactive["ci_hi"]],
            "reactive_signal": bool(reactive["ci_lo"] > 0),
            "planner_pilot_delta": planner["mean_delta"],
            "planner_pilot_ci": [planner["ci_lo"], planner["ci_hi"]],
            "planner_signal": bool(planner["ci_lo"] > 0),
            "boardkeep_wr": guard["wr"],
            "boardkeep_holds_parity": bool(guard["ci_hi"] < 0.5),
        },
    )

    log("=== E21 advanced-opponent zoo pilot DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
