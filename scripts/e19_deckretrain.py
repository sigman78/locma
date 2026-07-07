"""E19 deck-distribution retrain: close the train/deploy gap the promotion opened.

E18b's promotion swapped the draft half of both recipes of record to the
learned draft (depot:ldraft), but the b0k battle nets were trained entirely on
BALANCED decks (0.76 items/30, mean cost 4.8) and now deploy on ldraft decks
(4-5 items, cost ~3.3) -- the +0.117 reactive gain was earned DESPITE that
distribution gap. E19 retrains the battle recipe with the training env's decks
drafted by ldraft (draft_override plumbing: BothSeatsDraftPolicy keeps the two
seats' deck histories separate), so the net practices the decks it will pilot,
and its value head prices the positions those decks produce (E7: critics are
data-sensitive).

Treatment: the b0k recipe of record (token V0, lr=1e-4, target_kl=0.025,
5-phase zoo x 200k = 1M steps, cuda, n_envs=16) with
draft_override="depot:ldraft/ldraft_s{s}.zip" (matching seed). Everything
else unchanged. Models: runs/e19_b0kl_s{s}.zip ("b0k on ldraft decks").

Both sides of every verdict use the SAME ldraft draft half -- only the battle
net differs, isolating the retrain effect.

Stages (idempotent, resumable via runs/e19-summary.json):
  A. train b0kl_s{0,1,2} (sequential; ~35-50 min/seed).
  B. reactive ruler: [ppo:b0kl_sX,ldraft_sX] vs [ppo:depot:b0k_sX,ldraft_sX]
     (the current reactive RoR pair), 40x25 @ 20M anchors.
  C. planner arm, pilot-gated (critic read): [vbeam:b0kl_sX,8,20,ldraft_sX]
     vs [vbeam:depot:b0k_sX,8,20,ldraft_sX]; full 40x25 iff pilot
     mean_delta > 0. Informational for the deployed shared-ens RoR (E12:
     ensemble swaps need their own head-to-head), decisive for whether
     deck-matched training data improves the critic.
  D. confirm @ 21M fresh anchors iff stage B ci_lo > 0.
  E. boardkeep guard (E10 protocol, 20x50 mirrored @ 5M common random
     numbers): does the retrain hold or extend E18c's 0.408 neutralization?

Pre-registered gates:
  reactive promotion candidate iff B ci_lo > 0 AND D ci_lo > 0 (would make
  ppo:b0kl_sX,ldraft_sX the new reactive recipe of record).
  boardkeep guard flag iff its ci_lo > 0.5 (a regression past parity).

Seed ranges: eval anchors 20M primary / 21M confirm (1M-19M used through
E18b). Smoke: E19_SMOKE=1 -> tiny steps/grids, separate artifact paths.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E19_SMOKE") == "1"
WORKERS = 19
SEEDS = (0,) if SMOKE else (0, 1, 2)
SUMMARY_PATH = "runs/e19-smoke.json" if SMOKE else "runs/e19-summary.json"
LOG_PATH = "runs/e19-smoke.log" if SMOKE else "runs/e19.log"
MODEL_TMPL = "runs/e19-smoke_b0kl_s{s}.zip" if SMOKE else "runs/e19_b0kl_s{s}.zip"

STEPS_PER_OPP = 2_048 if SMOKE else 200_000
TRAIN_ENVS = 4 if SMOKE else 16
FULL = (2, 2) if SMOKE else (40, 25)  # (eval seeds, games_per_seed)
PILOT = (2, 2) if SMOKE else (10, 10)
PRIMARY_START = 20_000_000
CONFIRM_START = 21_000_000
GUARD_BLOCKS = 2 if SMOKE else 20
GUARD_GAMES = 2 if SMOKE else 50
GUARD_SEED0 = 5_000_000  # E10/E11/E18c common random numbers

B0K = [f"depot:b0k/b0k_s{s}.zip" for s in (0, 1, 2)]
LDRAFT = [f"depot:ldraft/ldraft_s{s}.zip" for s in (0, 1, 2)]

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


def pair(battle: str, s: int) -> str:
    """Reactive Composer spec: battle net + the seed's ldraft draft half."""
    return f"ppo:{battle},{LDRAFT[s]}"


def vpair(battle: str, s: int) -> str:
    return f"vbeam:{battle},8,20,{LDRAFT[s]}"


# ---- Stage A: training -------------------------------------------------------


def train_seed(s: int) -> None:
    """b0k recipe of record + draft_override to the matching ldraft seed."""
    from locma.envs.training import ZOO_OPPONENTS, train_zoo  # noqa: PLC0415

    out = model_path(s)
    if os.path.exists(out) and f"train_s{s}" in summary:
        log(f"train s{s}: exists, skip")
        return
    log(f"train s{s}: b0k recipe on ldraft decks, zoo {ZOO_OPPONENTS} -> {out}")
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
        draft_override=LDRAFT[s],  # the treatment; recipe otherwise = depot:b0k
    )
    record(
        f"train_s{s}",
        {
            "out": out,
            "draft_override": LDRAFT[s],
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


# ---- Stage E: boardkeep guard (E10/E11/E18c protocol) -------------------------


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

    defender = pair(model_path(0), 0)
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
            "e18c_b0k_ldraft_wr": 0.408,  # same seeds, pre-retrain pair
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E19 deck-distribution retrain start ===")

    # Stage A: train (sequential; each run owns the GPU + 16 subprocess envs).
    for s in SEEDS:
        train_seed(s)

    cands = [pair(model_path(s), s) for s in SEEDS]
    bases = [pair(B0K[s], s) for s in SEEDS]

    # Stage B: reactive ruler vs the current RoR pair (same drafts both sides).
    full = verdict("full_reactive", cands, bases, FULL, start=PRIMARY_START)

    # Stage C: planner/critic arm, pilot-gated, single critic per side.
    vb_cands = [vpair(model_path(s), s) for s in SEEDS]
    vb_bases = [vpair(B0K[s], s) for s in SEEDS]
    pilot = verdict("pilot_vbeam", vb_cands, vb_bases, PILOT, start=PRIMARY_START)
    if pilot["mean_delta"] > 0:
        verdict("full_vbeam", vb_cands, vb_bases, FULL, start=PRIMARY_START)
    else:
        record("full_vbeam_skipped", "pilot not positive")

    # Stage D: fresh-anchor confirm iff the primary ruler is CI-positive.
    if full["ci_lo"] > 0:
        verdict("confirm_reactive_21M", cands, bases, FULL, start=CONFIRM_START)

    # Stage E: boardkeep guard.
    boardkeep_guard()

    # Pre-registered gates.
    g: dict = {
        "reactive_delta": full["mean_delta"],
        "reactive_ci": [full["ci_lo"], full["ci_hi"]],
    }
    conf = summary.get("confirm_reactive_21M")
    g["promote_candidate"] = bool(full["ci_lo"] > 0 and conf is not None and conf["ci_lo"] > 0)
    if conf is not None:
        g["confirm_delta"] = conf["mean_delta"]
        g["confirm_ci"] = [conf["ci_lo"], conf["ci_hi"]]
    fv = summary.get("full_vbeam")
    g["critic_flag"] = bool(fv is not None and fv["ci_lo"] > 0)
    guard = summary["guard_boardkeep"]
    g["boardkeep_regression"] = bool(guard["ci_lo"] > 0.5)
    g["boardkeep_wr"] = guard["wr"]
    record("e19_gates", g)

    log("=== E19 deck-distribution retrain DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
