"""E26 — play-time micro-guards for the reactive rung (+ edraft full confirm).

Pre-registered 2026-07-13 (docs/e26-microguards-design.md), before any run.
Three zero-training levers, none touching a training loop:

  A. `lguard` — LethalGuardBattlePolicy (locma/policies/lguard.py): an
     exhaustive, cap-bounded own-turn DFS that plays a forced win when one
     exists this turn (E14a found the reactive net misses 18.3% of them) and
     otherwise gets out of the way. Spec `lppo:model,draft`.
  B. `ens` — MaskablePPOEnsembleBattlePolicy (locma/policies/ppo.py):
     mean-of-policy-heads over b0k_s0|s1|s2 (the POLICY-head analog of E8's
     mean-of-critics planner ensemble). Spec `ppo:s0|s1|s2,draft`.
  C. `lens` — both stacked: `lppo:s0|s1|s2,draft`.

Stage E is a pure-eval draft-side confirm (E20 open item): the zero-inference
`edraft` heuristic against the deployed `ldraft` halves, full scale directly
on BOTH rungs (reactive `ppo:` and planner `vbeam:`), no gate.

Protocol (E19 pattern): pilot 10x10 @ 34M per arm -> full 40x25 (same 34M
range) iff pilot ci_hi > 0 -> fresh confirm 40x25 @ 35M iff full ci_lo > 0.
Promotion candidate iff full AND confirm ci_lo > 0. Mechanism probe (serial,
50 games/opponent @ 36M): activations/game, guard-changed-the-move rate,
cap-hit rate, node counts, wall-clock overhead vs the plain RoR policy.

Stages, in order (each idempotent/resumable via runs/e26-summary.json):
  pilot_{a,b,c} -> full_{a,b,c} (gated) -> confirm_{a,b,c} (gated) ->
  probe_lguard (mechanism) -> stage_e_reactive -> stage_e_planner -> e26_gates.

Smoke: E26_SMOKE=1 -> tiny grids/games, separate `-smoke` artifact paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E26_SMOKE") == "1"
WORKERS = 19
SUMMARY_PATH = "runs/e26-summary-smoke.json" if SMOKE else "runs/e26-summary.json"
LOG_PATH = "runs/e26-smoke.log" if SMOKE else "runs/e26.log"

PILOT = (2, 2) if SMOKE else (10, 10)
FULL = (2, 2) if SMOKE else (40, 25)
PRIMARY_START = 34_000_000
CONFIRM_START = 35_000_000
PROBE_START = 36_000_000
PROBE_GAMES = 2 if SMOKE else 50

SEEDS = (0, 1, 2)
B0K = [f"depot:b0k/b0k_s{s}.zip" for s in SEEDS]
LDRAFT = [f"depot:ldraft/ldraft_s{s}.zip" for s in SEEDS]
EDRAFT = "depot:edraft/e20-elicit-fit.json"
SHARED_ENS = "depot:shared/shared_s0.zip|depot:shared/shared_s1.zip|depot:shared/shared_s2.zip"

BASELINES = [f"ppo:{B0K[s]},{LDRAFT[s]}" for s in SEEDS]  # reactive RoR pair (0.791)
ENS_BATTLE = "|".join(B0K)

ARM_A = [f"lppo:{B0K[s]},{LDRAFT[s]}" for s in SEEDS]
ARM_B = [f"ppo:{ENS_BATTLE},{LDRAFT[s]}" for s in SEEDS]
ARM_C = [f"lppo:{ENS_BATTLE},{LDRAFT[s]}" for s in SEEDS]

PROBE_OPPONENTS = ("scripted", "max-guard", "max-attack", "boardkeep")

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


# ---- Verdict wrapper (E19/E21 pattern) ---------------------------------------


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


def gated_stage(arm: str, candidates: list[str]) -> None:
    """pilot -> full (iff pilot ci_hi > 0) -> confirm (iff full ci_lo > 0)."""
    pilot = verdict(f"pilot_{arm}", candidates, BASELINES, PILOT, start=PRIMARY_START)
    if pilot["ci_hi"] <= 0:
        record(f"full_{arm}_skipped", "pilot ci_hi <= 0")
        return
    full = verdict(f"full_{arm}", candidates, BASELINES, FULL, start=PRIMARY_START)
    if full["ci_lo"] <= 0:
        record(f"confirm_{arm}_skipped", "full ci_lo <= 0")
        return
    verdict(f"confirm_{arm}", candidates, BASELINES, FULL, start=CONFIRM_START)


# ---- Mechanism probe (serial, no pool) ---------------------------------------


def probe_lguard() -> None:
    """Build the arm-A s0 policy directly (mirrors what the registry's
    ``lppo:`` factory does), play PROBE_GAMES vs each HARD3+boardkeep
    opponent, and record the guard's activation/change/cap-hit/node stats
    plus wall-clock overhead vs the plain (un-guarded) RoR policy."""
    tag = "probe_lguard"
    if tag in summary:
        log(f"{tag}: exists, skip")
        return
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.lguard import LethalGuardBattlePolicy  # noqa: PLC0415
    from locma.policies.ppo import MaskablePPOBattlePolicy, MaskablePPODraftPolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    b0k_s0 = resolve_path(B0K[0])
    ldraft_s0 = resolve_path(LDRAFT[0])

    guarded_battle = LethalGuardBattlePolicy(MaskablePPOBattlePolicy(model_path=b0k_s0), probe=True)
    guarded = Composer(
        guarded_battle, MaskablePPODraftPolicy(model_path=ldraft_s0), name="lguard-probe"
    )
    plain = make_policy(BASELINES[0])  # ppo:b0k_s0,ldraft_s0 (the un-guarded RoR)

    per_opponent: dict = {}
    guarded_seconds = 0.0
    plain_seconds = 0.0
    seed = PROBE_START
    for opp_name in PROBE_OPPONENTS:
        opp = make_policy(opp_name)
        t0 = time.time()
        res_g = run_match(guarded, opp, games=PROBE_GAMES, seed=seed)
        guarded_seconds += time.time() - t0
        opp2 = make_policy(opp_name)
        t0 = time.time()
        res_p = run_match(plain, opp2, games=PROBE_GAMES, seed=seed)
        plain_seconds += time.time() - t0
        per_opponent[opp_name] = {
            "guarded_wr": round(res_g.win_rate_a, 4),
            "plain_wr": round(res_p.win_rate_a, 4),
            "games": res_g.games,
        }
        seed += PROBE_GAMES
        log(f"probe vs {opp_name}: guarded={res_g.win_rate_a:.3f} plain={res_p.win_rate_a:.3f}")

    stats = guarded_battle.stats
    searches = max(stats["searches"], 1)
    activations = stats["activations"]
    total_games = PROBE_GAMES * len(PROBE_OPPONENTS) * 2  # run_match mirrors games x2

    record(
        tag,
        {
            "opponents": list(PROBE_OPPONENTS),
            "per_opponent": per_opponent,
            "stats": dict(stats),
            "activations_per_game": round(activations / max(total_games, 1), 4),
            "guard_changed_move_rate": round(stats["guard_changed_move"] / max(activations, 1), 4),
            "cap_hit_rate": round(stats["cap_hits"] / searches, 4),
            "mean_nodes_per_search": round(stats["nodes"] / searches, 2),
            "guarded_s_per_game": round(guarded_seconds / max(total_games, 1), 4),
            "plain_s_per_game": round(plain_seconds / max(total_games, 1), 4),
            "overhead_ratio": round(guarded_seconds / max(plain_seconds, 1e-9), 3),
            "total_games": total_games,
        },
    )


# ---- Stage E: edraft full confirm, both rungs, pure eval ---------------------


def stage_e() -> None:
    reactive_cands = [f"ppo:{B0K[s]},{EDRAFT}" for s in SEEDS]
    verdict("stage_e_reactive", reactive_cands, BASELINES, FULL, start=PRIMARY_START)

    planner_cands = [f"vbeam:{SHARED_ENS},8,20,{EDRAFT}"]
    planner_bases = [f"vbeam:{SHARED_ENS},8,20,{LDRAFT[s]}" for s in SEEDS]
    verdict("stage_e_planner", planner_cands, planner_bases, FULL, start=PRIMARY_START)


# ---- Gates --------------------------------------------------------------------


def _arm_gate(arm: str) -> dict:
    g: dict = {}
    pilot = summary.get(f"pilot_{arm}")
    full = summary.get(f"full_{arm}")
    confirm = summary.get(f"confirm_{arm}")
    if pilot is not None:
        g["pilot_delta"] = pilot["mean_delta"]
        g["pilot_ci"] = [pilot["ci_lo"], pilot["ci_hi"]]
    if full is not None:
        g["full_delta"] = full["mean_delta"]
        g["full_ci"] = [full["ci_lo"], full["ci_hi"]]
    if confirm is not None:
        g["confirm_delta"] = confirm["mean_delta"]
        g["confirm_ci"] = [confirm["ci_lo"], confirm["ci_hi"]]
    g["promote_candidate"] = bool(
        full is not None and full["ci_lo"] > 0 and confirm is not None and confirm["ci_lo"] > 0
    )
    g["headroom"] = bool(full is not None and full["mean_delta"] >= 0.03)
    return g


def gates() -> None:
    g: dict = {arm: _arm_gate(arm) for arm in ("a", "b", "c")}
    probe = summary.get("probe_lguard")
    if probe is not None:
        g["probe"] = {
            "activations_per_game": probe["activations_per_game"],
            "guard_changed_move_rate": probe["guard_changed_move_rate"],
            "cap_hit_rate": probe["cap_hit_rate"],
            "mean_nodes_per_search": probe["mean_nodes_per_search"],
            "overhead_ratio": probe["overhead_ratio"],
        }
    se_r = summary.get("stage_e_reactive")
    se_p = summary.get("stage_e_planner")
    if se_r is not None:
        g["stage_e_reactive_delta"] = se_r["mean_delta"]
        g["stage_e_reactive_ci"] = [se_r["ci_lo"], se_r["ci_hi"]]
    if se_p is not None:
        g["stage_e_planner_delta"] = se_p["mean_delta"]
        g["stage_e_planner_ci"] = [se_p["ci_lo"], se_p["ci_hi"]]
    record("e26_gates", g)


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E26 micro-guards start ===")

    # A/B/C pilots first (cheap), then fulls, then confirms (each gated stage
    # runs its own pilot->full->confirm ladder, in arm order).
    gated_stage("a", ARM_A)
    gated_stage("b", ARM_B)
    gated_stage("c", ARM_C)

    # Mechanism probe (serial, 200 games total across HARD3+boardkeep @ 36M).
    probe_lguard()

    # Stage E: edraft full confirm, both rungs, pure eval (no gate).
    stage_e()

    gates()
    log("=== E26 micro-guards DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
