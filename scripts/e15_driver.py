"""E15 driver: ranking-loss critic + AZ-v2 policy round (docs/e15-ranking-az2-design.md).

Stage 1 (load-bearing): can an ordering-aware objective transfer the 0.926
ensemble's sibling-ranking into ONE critic? Stage 2: does a listwise,
branching-weighted policy target land where E14a located the reactive
failure? Stage 3: iterate the passing stage once (critic side only -- the
policy head does not influence harvest targets, so re-harvesting for a
second policy round would train on identical labels).

Stages (idempotent -- each skips when its summary key / artifact exists):
  A. harvest: vbeam-on-ensemble self-play with sibling groups
     (collect_rank_data, ~150 games/opponent x 4 zoo opponents x 2 seats).
  B. Stage-1 FT x3 seeds: train_value_head_rank(depot:shared_sX) -> vrank_sX.
     G1 fidelity gate on pooled held-out pair accuracy.
     KILL: G1 fail -> record capacity verdict, skip C/D (per design).
  C. G2 primary: vbeam:vrank_sX vs vbeam:depot:shared_sX -- pilot 10x10
     (ci_hi > 0 gates the full), full 40x25 at 10M, confirm at 11M.
  D. G3 (iff G2 confirmed): vbeam:vrank singles vs the ensemble RoR at 10M
     (promotion = non-inferiority at 1/3 compute); bonus ens(vrank x3) arm.
  E. Stage-2 FT x3 seeds: train_policy_head_listwise(depot:shared_sX) ->
     vpi_sX (policy branch only; critic path byte-identical).
  F. G4 reactive: vpi_sX vs depot:b0k full at 10M, confirm 11M (the
     registered absorption bar); informational arm vs the shared_sX bases
     (clean paired-base delta).
  G. G6 planner side-effects: vbeam:vpi_sX vs vbeam:shared_sX full at 10M;
     non-inferiority = ci_hi > -0.01 (only would_pass can shift).
  H. G5 mechanism probes (informational): compact E14a-style shadow probes
     (root disagreement, missed lethal, item underuse) on shared_s0 (base)
     vs vpi_s0, 50 seeds x 2 seats x HARD3+boardkeep each.
  I. Stage 3 (iff G2 passed): re-harvest with the vrank ensemble as
     evaluator, FT round 2 -> vrank2_sX, adopt iff full vbeam:vrank2 vs
     vbeam:vrank at 12M has ci_lo > 0.

Pre-registered gates (design doc):
  G1: pooled val pair accuracy a1 >= a0 + 0.5*(1 - a0).
  G2: full AND confirm ci_lo > 0.
  G3: promote iff G2 AND vs-ensemble ci_hi >= 0 (CI contains or exceeds 0).
  G4: full AND confirm ci_lo > 0 (vs depot:b0k).
  G6: ci_hi > -0.01.
Pre-registered constants: RANK_TEMP=0.05, ANCHOR_LAMBDA=0.25,
MIN_MARGIN=0.01, MAX_PAIRS_PER_GROUP=15, LISTWISE_TAU=0.05 (vbeam_rank.py);
FT epochs=10, lr=3e-4 (E9 parity -- the objective is the only moved
variable).

Seed ledger: harvest seed 0 (training-side); eval 10M primary / 11M confirm
/ 12M stage-3 (1M-9M spent by earlier experiments).

Progress in runs/e15-overnight.log, results in runs/e15-summary.json.
Smoke mode: E15_SMOKE=1 -> tiny grids, separate paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E15_SMOKE") == "1"
WORKERS = 4 if SMOKE else 19
SUMMARY_PATH = "runs/e15-smoke.json" if SMOKE else "runs/e15-summary.json"
LOG_PATH = "runs/e15-smoke.log" if SMOKE else "runs/e15-overnight.log"
DATA_PATH = "runs/e15-smoke-rankdata.npz" if SMOKE else "runs/e15-rankdata.npz"
DATA2_PATH = "runs/e15-smoke-rankdata2.npz" if SMOKE else "runs/e15-rankdata2.npz"
VRANK_TMPL = "runs/e15-smoke-vrank_s{s}.zip" if SMOKE else "runs/vrank_s{s}.zip"
VRANK2_TMPL = "runs/e15-smoke-vrank2_s{s}.zip" if SMOKE else "runs/vrank2_s{s}.zip"
VPI_TMPL = "runs/e15-smoke-vpi_s{s}.zip" if SMOKE else "runs/vpi_s{s}.zip"

HARVEST_GAMES = 2 if SMOKE else 150  # per opponent; x4 opponents x2 seats
FT_EPOCHS = 2 if SMOKE else 10
PILOT = (2, 2) if SMOKE else (10, 10)
FULL = (2, 2) if SMOKE else (40, 25)
PROBE_SEEDS = 2 if SMOKE else 50
PRIMARY_START = 10_000_000
CONFIRM_START = 11_000_000
STAGE3_START = 12_000_000

ALL_SEEDS = (0, 1, 2)
SEEDS = (0,) if SMOKE else ALL_SEEDS
SHARED = [f"depot:shared/shared_s{s}.zip" for s in ALL_SEEDS]
B0K = [f"depot:b0k/b0k_s{s}.zip" for s in ALL_SEEDS]
ENS_ROR = "vbeam:" + "|".join(SHARED)
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


def _round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


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


# ---------------------------------------------------------------------------
# Stage A: harvest
# ---------------------------------------------------------------------------


def harvest(tag: str, model_paths: list[str], out_path: str) -> None:
    from locma.envs.vbeam_rank import collect_rank_data  # noqa: PLC0415

    if tag in summary and os.path.exists(out_path):
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    manifest = collect_rank_data(
        model_paths, out_path, games=HARVEST_GAMES, seed=0, workers=WORKERS
    )
    manifest["minutes"] = round((time.time() - t0) / 60, 1)
    record(tag, _round_floats({k: v for k, v in manifest.items() if k != "model_paths"}))


# ---------------------------------------------------------------------------
# Stage B/E: fine-tunes
# ---------------------------------------------------------------------------


def ft_rank(tag: str, base: str, data: str, out_path: str, seed: int) -> None:
    from locma.envs.vbeam_rank import train_value_head_rank  # noqa: PLC0415

    if tag in summary and os.path.exists(out_path):
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    m = train_value_head_rank(base, data, out_path, epochs=FT_EPOCHS, seed=seed)
    m["minutes"] = round((time.time() - t0) / 60, 1)
    record(tag, _round_floats(m))


def ft_listwise(tag: str, base: str, data: str, out_path: str, seed: int) -> None:
    from locma.envs.vbeam_rank import train_policy_head_listwise  # noqa: PLC0415

    if tag in summary and os.path.exists(out_path):
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    m = train_policy_head_listwise(base, data, out_path, epochs=FT_EPOCHS, seed=seed)
    m["minutes"] = round((time.time() - t0) / 60, 1)
    record(tag, _round_floats(m))


def g1_gate(prefix: str) -> bool:
    key = f"g1_{prefix}"
    if key in summary:
        return summary[key]["pass"]
    a0s = [summary[f"{prefix}_s{s}"]["before"]["acc"] for s in SEEDS]
    a1s = [summary[f"{prefix}_s{s}"]["after"]["acc"] for s in SEEDS]
    a0, a1 = sum(a0s) / len(a0s), sum(a1s) / len(a1s)
    bar = a0 + 0.5 * (1.0 - a0)
    record(
        key,
        {
            "pass": bool(a1 >= bar),
            "acc_before": round(a0, 4),
            "acc_after": round(a1, 4),
            "bar": round(bar, 4),
        },
    )
    return summary[key]["pass"]


# ---------------------------------------------------------------------------
# Stage H: compact mechanism probes (E14a headline metrics)
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _probe_game(model_ref: str, opp_spec: str, seed: int, seat: int) -> tuple:
    """One shadow game; returns (root_n, root_agree, planner_U, net_U,
    lethal_turns, missed_lethal)."""
    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.actions import Pass, Use  # noqa: PLC0415
    from locma.core.engine import make_battle_view, run_game  # noqa: PLC0415
    from locma.core.state import Phase  # noqa: PLC0415
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.encode import sem_index  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415
    from locma.policies.vbeam import NetValueEvaluator, _clone_battle, plan_turn  # noqa: PLC0415

    key = f"net:{model_ref}"
    if key not in _CACHE:
        _CACHE[key] = (
            make_policy(f"ppo:{model_ref}"),
            NetValueEvaluator(resolve_path(model_ref)),
        )
    if opp_spec not in _CACHE:
        _CACHE[opp_spec] = make_policy(opp_spec)
    net, ev = _CACHE[key]
    opp = _CACHE[opp_spec]

    c = {"root_n": 0, "root_agree": 0, "pl_u": 0, "net_u": 0, "lethal": 0, "missed": 0}
    st = {"turn": -1, "plan_win": False}

    def _plan_wins(gs, plan) -> bool:
        sim = _clone_battle(gs)
        for a in plan:
            if isinstance(a, Pass):
                return False
            battlemod.apply_battle(sim, a)
            if sim.phase == Phase.ENDED:
                return sim.winner == seat
        return False

    def hook(s: int, action, gs) -> None:
        if s != seat or gs.phase != Phase.BATTLE:
            return
        legal = list(battlemod.battle_legal(gs))
        if len(legal) < 2:
            return
        new_turn = gs.turn != st["turn"]
        if new_turn:
            # Reaching a NEW own turn means the game did not end on the
            # previous one: a plan-win there was left unconverted.
            if st["plan_win"]:
                c["missed"] += 1
            st["turn"] = gs.turn
            st["plan_win"] = False
        plan = plan_turn(gs, ev, width=8)
        view = make_battle_view(gs)
        if new_turn:
            c["root_n"] += 1
            c["root_agree"] += sem_index(view, action) == sem_index(view, plan[0])
            if _plan_wins(gs, plan):
                c["lethal"] += 1
                st["plan_win"] = True
        c["pl_u"] += isinstance(plan[0], Use)
        c["net_u"] += isinstance(action, Use)

    if seat == 0:
        res = run_game(net, opp, seed, on_pre_step=hook)
    else:
        res = run_game(opp, net, seed, on_pre_step=hook)
    # Final turn: a plan-win converts iff the game ended with our win.
    if st["plan_win"] and res.winner != seat:
        c["missed"] += 1
    return (c["root_n"], c["root_agree"], c["pl_u"], c["net_u"], c["lethal"], c["missed"])


def probes(ex, tag: str, model_ref: str) -> None:
    if tag in summary:
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    specs = [
        (model_ref, opp, PRIMARY_START + i, seat)
        for opp in PROBE_OPPONENTS
        for i in range(PROBE_SEEDS)
        for seat in (0, 1)
    ]
    totals = [0] * 6
    for r in ex.map(_probe_game, *zip(*specs, strict=True)):
        for i, v in enumerate(r):
            totals[i] += v
    root_n, root_agree, pl_u, net_u, lethal, missed = totals
    record(
        tag,
        {
            "games": len(specs),
            "root_disagree": round(1 - root_agree / max(1, root_n), 4),
            "planner_use_actions": pl_u,
            "net_use_actions": net_u,
            "item_underuse_ratio": round(pl_u / max(1, net_u), 2),
            "plan_win_turns": lethal,
            "missed_lethal_rate": round(missed / max(1, lethal), 4),
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gates() -> None:
    if "e15_gates" in summary:
        log("e15_gates: exists, skip")
        return
    g: dict = {}
    g1 = summary.get("g1_fidelity", {})
    if not g1.get("pass"):
        g["stage1"] = "G1 FAIL: frozen extractor cannot represent the ensemble ordering"
    else:
        full = summary.get("full_g2")
        conf = summary.get("confirm_g2_11M")
        if full is None:
            g["stage1"] = "G1 pass; G2 pilot-gated out"
        elif full["ci_lo"] <= 0:
            g["stage1"] = f"G1 pass; G2 full null ({full['mean_delta']}) -- direction closed"
        elif conf is None or conf["ci_lo"] <= 0:
            g["stage1"] = "G2 full CI-positive but unconfirmed"
        else:
            ens = summary.get("g3_vs_ens", {})
            promote = ens and ens.get("ci_hi", -1) >= 0
            g["stage1"] = f"G2 CONFIRMED (+{conf['mean_delta']}); " + (
                "G3 PROMOTE at 1/3 compute" if promote else "G3 below ensemble"
            )
    g4 = summary.get("full_g4")
    g4c = summary.get("confirm_g4_11M")
    if g4 is None:
        g["stage2_g4"] = "not run"
    elif g4["ci_lo"] <= 0:
        g["stage2_g4"] = f"null vs b0k ({g4['mean_delta']})"
    elif g4c is None or g4c["ci_lo"] <= 0:
        g["stage2_g4"] = "full CI-positive but unconfirmed"
    else:
        g["stage2_g4"] = f"ABSORPTION CONFIRMED (+{g4c['mean_delta']})"
    g6 = summary.get("full_g6")
    if g6 is not None:
        g["stage2_g6"] = "non-inferior" if g6["ci_hi"] > -0.01 else f"REGRESSION ({g6['ci_hi']})"
    s3 = summary.get("full_stage3_12M")
    if s3 is not None:
        g["stage3"] = f"iter2 {'ADOPT' if s3['ci_lo'] > 0 else 'no gain'} ({s3['mean_delta']})"
    else:
        g["stage3"] = "not triggered"
    record("e15_gates", g)


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E15 driver start ===")

    # Stage A: harvest with the ensemble RoR as evaluator.
    harvest("harvest", SHARED, DATA_PATH)

    # Stage B: ranking-critic FTs + G1.
    for s in SEEDS:
        ft_rank(f"fidelity_s{s}", SHARED[s], DATA_PATH, VRANK_TMPL.format(s=s), s)
    g1 = g1_gate("fidelity")

    vrank = [VRANK_TMPL.format(s=s) for s in SEEDS]
    vb_vrank = [f"vbeam:{p}" for p in vrank]
    vb_shared = [f"vbeam:{p}" for p in SHARED[: len(SEEDS)]]

    # Stage C: G2 primary (killed by G1 failure per design).
    if g1:
        pilot = verdict("pilot_g2", vb_vrank, vb_shared, PILOT, start=PRIMARY_START)
        if pilot["ci_hi"] > 0:
            full = verdict("full_g2", vb_vrank, vb_shared, FULL, start=PRIMARY_START)
            if full["ci_lo"] > 0:
                verdict("confirm_g2_11M", vb_vrank, vb_shared, FULL, start=CONFIRM_START)
        else:
            log("full_g2: pilot ci_hi <= 0, not promoted")
        # Stage D: G3 iff confirmed.
        conf = summary.get("confirm_g2_11M")
        if conf and conf["ci_lo"] > 0:
            verdict("g3_vs_ens", vb_vrank, [ENS_ROR], FULL, start=PRIMARY_START)
            ens_vrank = "vbeam:" + "|".join(vrank)
            verdict("g3_ens3", [ens_vrank], [ENS_ROR], FULL, start=PRIMARY_START)
    else:
        log("stage C/D: killed by G1 failure (capacity verdict)")

    # Stage E: listwise policy FTs.
    for s in SEEDS:
        ft_listwise(f"stage2_s{s}", SHARED[s], DATA_PATH, VPI_TMPL.format(s=s), s)
    vpi = [VPI_TMPL.format(s=s) for s in SEEDS]

    # Stage F: G4 reactive (registered bar: vs depot:b0k) + paired-base arm.
    full4 = verdict("full_g4", vpi, B0K[: len(SEEDS)], FULL, start=PRIMARY_START)
    if full4["ci_lo"] > 0:
        verdict("confirm_g4_11M", vpi, B0K[: len(SEEDS)], FULL, start=CONFIRM_START)
    verdict("full_g4_vs_base", vpi, SHARED[: len(SEEDS)], FULL, start=PRIMARY_START)

    # Stage G: G6 planner side-effects.
    verdict("full_g6", [f"vbeam:{p}" for p in vpi], vb_shared, FULL, start=PRIMARY_START)

    # Stage H: mechanism probes (base vs stage-2 net).
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_eval_worker) as ex:
        probes(ex, "probes_base", SHARED[0])
        probes(ex, "probes_vpi", vpi[0])

    # Stage I: one critic iteration iff G2 confirmed.
    conf = summary.get("confirm_g2_11M")
    if g1 and conf and conf["ci_lo"] > 0:
        harvest("harvest2", vrank, DATA2_PATH)
        for s in SEEDS:
            ft_rank(f"fidelity2_s{s}", vrank[s], DATA2_PATH, VRANK2_TMPL.format(s=s), s)
        vb_vrank2 = [f"vbeam:{VRANK2_TMPL.format(s=s)}" for s in SEEDS]
        verdict("full_stage3_12M", vb_vrank2, vb_vrank, FULL, start=STAGE3_START)
    else:
        log("stage I: not triggered (G2 not confirmed)")

    gates()
    log("=== E15 driver DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
