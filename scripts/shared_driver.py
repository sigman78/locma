"""E7 shared-draft study driver: what does the shared draft variant change?

The shared variant (PR feat/shared-draft): a pick removes the card from the
other seat's offer, the second picker chooses from the remaining 2, first
pick alternates by round. It breaks the default rule's mirror (same
deterministic draft on both seats = identical decks).

Stages (idempotent — each skips when its summary key already exists):
  A. draft-bench round robin (7 built-in drafts), default vs shared,
     under the ground pilot and the deployment b0 net pilot
  B. policy-roster tournament (matrix + Elo), default vs shared
  C. train 3 seeds of the B0 recipe of record + shared_draft=True
     (token V0, lr=1e-4, target_kl=0.025, 800k zoo, n_envs=16, cuda)
  D. paired ceiling-eval verdicts:
       reactive shared_sX vs depot:b0  on the standard ruler
       reactive shared_sX vs depot:b0  under shared-draft deployment
       vbeam:shared_sX  vs vbeam:b0    on the standard ruler   (pilot-gated)
       vbeam:shared_sX  vs vbeam:b0    under shared deployment (pilot-gated)

Artifacts land in runs/ (shared_s*.zip), progress in runs/shared-overnight.log,
machine-readable results in runs/shared-summary.json (rewritten after every
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
DRAFTS = ("random", "greedy", "weighted", "max-attack", "max-defense", "max-guard", "balanced")
ROSTER = (
    "random",
    "scripted",
    "greedy",
    "max-guard",
    "max-attack",
    "dmcts",
    "ppo:depot:b0/b0_s0.zip",
    "vbeam:depot:b0/b0_s0.zip",
)
SUMMARY_PATH = "runs/shared-summary.json"
LOG_PATH = "runs/shared-overnight.log"

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


def _matrix_json(win_matrix: dict) -> dict:
    """Tuple-keyed win matrix -> 'a|b'-keyed (JSON-serializable), rounded."""
    return {f"{a}|{b}": round(r, 4) for (a, b), r in win_matrix.items()}


def bench(tag: str, battle: str, games: int, shared: bool) -> None:
    from locma.harness.draft_bench import round_robin  # noqa: PLC0415 — lazy heavy import

    if tag in summary:
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    s = round_robin(
        list(DRAFTS), battle=battle, games=games, seed=0, workers=WORKERS, shared=shared
    )
    record(
        tag,
        {
            "battle": battle,
            "games_per_pair": 2 * games,
            "shared": shared,
            "avg_win_rate": {k: round(v, 4) for k, v in s.avg_win_rate.items()},
            "win_matrix": _matrix_json(s.win_matrix),
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


def tourney(tag: str, games: int, shared: bool) -> None:
    from locma.harness.tournament import run_tournament  # noqa: PLC0415 — lazy heavy import
    from locma.policies.registry import make_policy  # noqa: PLC0415

    if tag in summary:
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    pols = [make_policy(n) for n in ROSTER]
    res = run_tournament(pols, games=games, seed=0, reference="random", shared_draft=shared)
    record(
        tag,
        {
            "games_per_pair": 2 * games,
            "shared": shared,
            "elo": {k: round(v, 1) for k, v in res.ratings.items()},
            "win_matrix": _matrix_json(res.win_matrix),
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


def train_seed(s: int) -> None:
    """B0 recipe of record (see depot:b0 provenance) + shared_draft=True."""
    from locma.envs.training import train_zoo  # noqa: PLC0415 — lazy heavy import

    out = f"runs/shared_s{s}.zip"
    if os.path.exists(out) and f"train_s{s}" in summary:
        log(f"train s{s}: exists, skip")
        return
    log(f"train s{s}: B0 recipe + shared draft -> {out}")
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
    )
    record(f"train_s{s}", {"out": out, "minutes": round((time.time() - t0) / 60, 1)})


def verdict(
    tag: str, candidates: list[str], baselines: list[str], seeds: int, games: int, shared: bool
) -> dict:
    from locma.harness.ceiling_eval import (  # noqa: PLC0415 — lazy heavy import
        _disjoint_eval_seeds,
        run_verdict,
    )

    if tag in summary:
        log(f"{tag}: exists, skip")
        return summary[tag]
    t0 = time.time()
    out = run_verdict(
        candidates,
        baselines,
        seeds=_disjoint_eval_seeds(seeds, games),
        games_per_seed=games,
        workers=WORKERS,
        shared_draft=shared,
    )
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    out["shared_deploy"] = shared
    out["minutes"] = round((time.time() - t0) / 60, 1)
    record(tag, out)
    return out


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E7 shared-draft study driver start ===")

    # ---- Stage A: draft-bench, default vs shared --------------------------
    bench("bench_ground_default", "ground", games=300, shared=False)
    bench("bench_ground_shared", "ground", games=300, shared=True)
    bench("bench_b0_default", "ppo:depot:b0/b0_s0.zip", games=150, shared=False)
    bench("bench_b0_shared", "ppo:depot:b0/b0_s0.zip", games=150, shared=True)

    # ---- Stage B: policy tournament, default vs shared ---------------------
    tourney("tournament_default", games=50, shared=False)
    tourney("tournament_shared", games=50, shared=True)

    # ---- Stage C: train the shared-draft arm (B0 recipe + shared) ----------
    for s in SEEDS:
        train_seed(s)

    # ---- Stage D: paired verdicts ------------------------------------------
    cands = [f"runs/shared_s{s}.zip" for s in SEEDS]
    bases = [b0(s) for s in SEEDS]

    # Reactive, standard ruler (transfer to the default game).
    verdict("full_reactive_standard", cands, bases, seeds=40, games=25, shared=False)
    # Reactive, shared-draft deployment (home turf: does the shared-trained
    # net win the shared game? b0_avg here is B0's number under shared play).
    verdict("full_reactive_shared_deploy", cands, bases, seeds=40, games=25, shared=True)

    # Planner arms (vbeam is ~0.6 s/game — pilot-gate the full runs).
    vb_cands = [f"vbeam:runs/shared_s{s}.zip" for s in SEEDS]
    vb_bases = [f"vbeam:{b0(s)}" for s in SEEDS]
    for mode, shared in (("standard", False), ("shared_deploy", True)):
        pilot = verdict(
            f"pilot_vbeam_{mode}", vb_cands, vb_bases, seeds=10, games=10, shared=shared
        )
        if pilot["mean_delta"] > -0.10:
            verdict(f"full_vbeam_{mode}", vb_cands, vb_bases, seeds=40, games=25, shared=shared)
        else:
            record(f"full_vbeam_{mode}_skipped", "pilot clearly negative (< -0.10)")

    log("=== E7 shared-draft study driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
