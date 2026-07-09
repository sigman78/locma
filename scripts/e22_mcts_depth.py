"""E22 pilot: how deep would cheating MCTS (and fair dMCTS) need to search to
beat the current strongest policy, at a matched ("same") draft?

Motivation. Every `mcts:100`/`dmcts:15,30` number on file predates the current
planner recipe of record: `mcts:100` beat the ~June-era reactive PPO (~0.55
avg-hard3) about 80% of the time and was roughly even with `azlite` (0.57).
Since then the planner moved to `vbeam:<3-critic ensemble>,8,20,ldraft`
(0.978 avg-hard3, confirmed RoR, docs/baseline.md "2026-07-07"). Cheating
MCTS's edge is an INFORMATION advantage (it clones the real GameState --
perfect knowledge of the opponent's hand); the planner's edge is a STRUCTURAL
one (a trained value critic ensemble replacing random/heuristic rollouts).
Whether more search depth lets the information advantage catch up to the
structural one at today's strength is an empirical question, not a "MCTS
always wins eventually" theorem -- UCT's regret bound assumes a stationary
leaf evaluator; a heuristic-rollout leaf estimate does not become a better
value function just because you sample it more.

This required one piece of plumbing: `mcts:`/`dmcts:` registry specs hardcoded
`GreedyDraftPolicy` (no draft-override param, unlike `ppo:`/`vbeam:`). Added a
5th positional param to both (locma/policies/registry.py, `_greedy_draft_param`
-- defaults to `GreedyDraftPolicy` so every historical bare `mcts:100` spec
stays byte-identical; only opts into an override when the param is given).

Design. Direct head-to-head (run_match, mirrored, Wilson 95% CI) -- NOT the
avg-hard3-vs-HARD3-pool ruler every other experiment here uses, because the
question is "who wins when they actually play each other", not "who beats a
common reference pool better" (ratings/pool deltas can mislead on
non-transitive matchups -- see docs/baseline.md's "read the matrix" note).
Both sides use the SAME draft half: `depot:ldraft/ldraft_s0.zip` (E18b, the
confirmed reactive+planner RoR draft, not the pilot-scale `edraft` heuristic)
-- isolates search quality as the only variable.

  candidate  (our best): vbeam:<3-critic shared ensemble>,8,20,ldraft_s0
  cheating MCTS sweep:   mcts:{100,1000,5000,20000},sqrt2,0,3,ldraft_s0
  fair dMCTS sweep:      dmcts:15,{30,100,500,2000},0,3,ldraft_s0
    (K=15 fixed at the existing historical default; I swept so the K*I total
    simulation budget brackets the same order of magnitude as the mcts sweep
    -- I=30 reproduces the codebase's prior "dmcts ~ as strong as cheating
    mcts" reference point as the sweep's low anchor)

Both search families are pure CPU (heuristic rollout, no net) except the
candidate side, which loads the 3-critic ensemble + the ldraft net once per
worker (cached, matching the `_cell` pattern in scripts/e19_deckretrain.py /
e21_advopponent.py).

This is a PILOT: 100 games/cell (10 blocks x 10, Wilson CI), no full grid, no
promotion gate -- it answers "roughly what depth, if any, closes the gap"
and whether dMCTS behaves differently from cheating MCTS at matched budget.

Seed range: SEED0 = 26_000_000 (fresh, after E21's 24M/25M; head-to-head
cells use run_match's own mirrored-block semantics, not the disjoint-eval-
seed ruler, so exact spacing matters less than staying out of used ranges).
Smoke: E22_SMOKE=1 -> tiny iteration counts/games, separate artifact paths.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E22_SMOKE") == "1"
WORKERS = 19
SUMMARY_PATH = "runs/e22-smoke.json" if SMOKE else "runs/e22-summary.json"
LOG_PATH = "runs/e22-smoke.log" if SMOKE else "runs/e22.log"

BLOCKS = 2 if SMOKE else 10
GAMES = 2 if SMOKE else 10  # per block; mirrored internally by run_match
SEED0 = 26_000_000

SHARED = [f"depot:shared/shared_s{s}.zip" for s in (0, 1, 2)]
LDRAFT0 = "depot:ldraft/ldraft_s0.zip"
C_UCT = math.sqrt(2)
ROLLOUT_TURNS = 3
DMCTS_K = 15

# The candidate: the confirmed planner RoR ensemble, paired with the RoR draft.
ENS = "vbeam:" + "|".join(SHARED) + f",8,20,{LDRAFT0}"

MCTS_ITERS = (10, 20) if SMOKE else (100, 1000, 5000, 20000)
DMCTS_ITERS = (10, 20) if SMOKE else (30, 100, 500, 2000)  # per-world I; K fixed


def mcts_spec(iters: int) -> str:
    return f"mcts:{iters},{C_UCT},0,{ROLLOUT_TURNS},{LDRAFT0}"


def dmcts_spec(i: int) -> str:
    return f"dmcts:{DMCTS_K},{i},0,{ROLLOUT_TURNS},{LDRAFT0}"


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


# ---- head-to-head cells (E10/E11/E19/E21 protocol) ---------------------------

_CACHE: dict = {}


def _cell(a: str, b: str, seed: int, games: int) -> tuple[int, int]:
    """Picklable pool unit: one seed block of run_match(a, b). a's win count first."""
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    for spec in (a, b):
        if spec not in _CACHE:
            _CACHE[spec] = make_policy(spec)
    res = run_match(_CACHE[a], _CACHE[b], games=games, seed=seed)
    return res.wins_a, res.games


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def head_to_head(ex, tag: str, a: str, b: str) -> dict:
    if tag in summary:
        log(f"{tag}: exists, skip")
        return summary[tag]
    t0 = time.time()
    seeds = [SEED0 + blk * GAMES for blk in range(BLOCKS)]
    results = list(ex.map(_cell, [a] * BLOCKS, [b] * BLOCKS, seeds, [GAMES] * BLOCKS))
    wins = sum(w for w, _ in results)
    n = sum(g for _, g in results)
    wr, lo, hi = _wilson(wins, n)
    out = {
        "a": a,
        "b": b,
        "a_wr": round(wr, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "games": n,
        "b_wins": bool(hi < 0.5),  # candidate's ci_hi < 0.5 -> the searcher is CI-ahead
        "minutes": round((time.time() - t0) / 60, 1),
    }
    record(tag, out)
    return out


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E22 MCTS-depth pilot start ===")
    log(f"candidate (our best): {ENS}")

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_eval_worker) as ex:
        mcts_rows = [head_to_head(ex, f"mcts_{n}", ENS, mcts_spec(n)) for n in MCTS_ITERS]
        dmcts_rows = [head_to_head(ex, f"dmcts_{i}", ENS, dmcts_spec(i)) for i in DMCTS_ITERS]

    read = {
        "mcts": [
            {
                "iters": n,
                "candidate_wr": r["a_wr"],
                "ci": [r["ci_lo"], r["ci_hi"]],
                "searcher_ahead": r["b_wins"],
            }
            for n, r in zip(MCTS_ITERS, mcts_rows, strict=True)
        ],
        "dmcts": [
            {
                "K": DMCTS_K,
                "I": i,
                "total_sims": DMCTS_K * i,
                "candidate_wr": r["a_wr"],
                "ci": [r["ci_lo"], r["ci_hi"]],
                "searcher_ahead": r["b_wins"],
            }
            for i, r in zip(DMCTS_ITERS, dmcts_rows, strict=True)
        ],
    }
    record("e22_read", read)
    log("=== E22 MCTS-depth pilot DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
