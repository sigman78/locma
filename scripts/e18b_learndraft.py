"""E18b learned draft: can RL beat the hand-tuned balanced draft?

The staged, single-box version of ByteRL's end-to-end draft+battle training
(arXiv 2303.04096 -- the COG 2022 winner's one distinctive ingredient we have
never tried; see E18a for the gamma lever from the same paper). Every draft we
deploy is the hand-written BalancedDraftPolicy; E17 established only that the
best member of that scripted-scorer FAMILY is the current one. This trains a
draft net by RL against a FROZEN battle pilot (DraftEnv: mirror pilot on both
seats, so the terminal reward isolates deck quality) and asks whether learned
deck-building beats the scripted scorer at all.

Training: draft_s{0,1,2}, each against its matching reactive pilot
depot:b0k/b0k_s{s} (the reactive recipe of record) vs a balanced-drafting
opponent -- optimizing exactly the G1 matchup. rollouts=3 (reward = mean win
of 3 reshuffled playouts, variance/3 on the 30-picks-one-outcome credit
assignment), 300k picks = 10k episodes, gamma=1.0 (terminal-only reward),
n_envs=16. ~35 min/seed at the measured ~1s/episode/env (x3 rollouts).

Stages (idempotent, resumable via runs/e18b-summary.json):
  A. train draft_s{0,1,2} -> runs/e18b_draft_s{s}.zip.
  B. census: 200 seeded learned-vs-balanced drafts -> items/deck, mean cost,
     curve; degenerate policies (all-same-pick) show up here. Informational.
  C. G1 reactive (the KILL gate): [ppo:b0k_sX,draft_sX] vs [ppo:b0k_sX],
     pilot 10x10 @ 18M; full 40x25 iff pilot mean_delta > 0. The learned
     draft was trained for THIS pilot -- if it cannot beat balanced here,
     the draft seam closes the way E17 closed the scripted-scorer axis.
  D. G2 planner: [vbeam:shared_ens,8,20,draft_sX] vs the 0.926 RoR, same
     pilot->full gating. Independent read: the draft is reactive-tuned, but
     the planner converts items ~1.7x better (E16a), so a deck the reactive
     pilot merely tolerates may score under the planner.
  E. G3 confirm @ 19M fresh anchors for any full arm with ci_lo > 0.

Pre-registered gates:
  draft_seam_open iff any full arm ci_lo > 0 AND its confirm ci_lo > 0.
  Planner promotion additionally needs the standard +0.03 headroom bar
  (run_verdict's own threshold) -- recorded, not auto-claimed.
  If G1 pilot fails on all seeds, the reactive arm is dead; G2 pilot still
  runs (cheap, pre-registered as independent), full only on a positive pilot.

Run AFTER E18a's box time frees up (both want the GPU + 16 subprocess envs).
Seed ranges: eval anchors 18M primary / 19M confirm (1M-17M used through
E18a). Training env seeds stay in 0..800k, below the 1M+ eval range.
Smoke: E18B_SMOKE=1 -> tiny steps/grids, separate artifact paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E18B_SMOKE") == "1"
WORKERS = 19
SEEDS = (0,) if SMOKE else (0, 1, 2)
SUMMARY_PATH = "runs/e18b-smoke.json" if SMOKE else "runs/e18b-summary.json"
LOG_PATH = "runs/e18b-smoke.log" if SMOKE else "runs/e18b.log"
DRAFT_TMPL = "runs/e18b-smoke_draft_s{s}.zip" if SMOKE else "runs/e18b_draft_s{s}.zip"

TRAIN_STEPS = 480 if SMOKE else 300_000
TRAIN_ENVS = 2 if SMOKE else 16
# SB3 collects a full n_steps-per-env rollout buffer before the first update,
# so the smoke needs a small n_steps or its "480-step" train runs 4096 steps.
TRAIN_N_STEPS = 120 if SMOKE else 2048
ROLLOUTS = 1 if SMOKE else 3
CENSUS_N = 20 if SMOKE else 200
PILOT = (2, 2) if SMOKE else (10, 10)  # (eval seeds, games_per_seed)
FULL = (2, 2) if SMOKE else (40, 25)
PRIMARY_START = 18_000_000
CONFIRM_START = 19_000_000

B0K = [f"depot:b0k/b0k_s{s}.zip" for s in (0, 1, 2)]
SHARED = [f"depot:shared/shared_s{s}.zip" for s in (0, 1, 2)]
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


def draft_path(s: int) -> str:
    return DRAFT_TMPL.format(s=s)


def ens_spec(s: int) -> str:
    return "vbeam:" + "|".join(SHARED) + f",8,20,{draft_path(s)}"


# ---- Stage A: train the draft nets ------------------------------------------


def train_seed(s: int) -> None:
    from locma.envs.training import train_draft  # noqa: PLC0415

    out = draft_path(s)
    if os.path.exists(out) and f"train_s{s}" in summary:
        log(f"train s{s}: exists, skip")
        return
    log(f"train s{s}: draft net vs balanced, pilot {B0K[s]} -> {out}")
    t0 = time.time()
    train_draft(
        B0K[s],  # frozen pilot: the matching reactive recipe-of-record seed
        steps=TRAIN_STEPS,
        out=out,
        seed=s,
        opponent_draft="balanced",  # the incumbent the learned draft must beat
        rollouts=ROLLOUTS,
        n_envs=TRAIN_ENVS,
        n_steps=TRAIN_N_STEPS,
        verbose=0,
    )
    record(
        f"train_s{s}",
        {
            "out": out,
            "pilot": B0K[s],
            "steps": TRAIN_STEPS,
            "rollouts": ROLLOUTS,
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


# ---- Stage B: draft census (learned vs balanced decks) ----------------------


def census() -> None:
    if "census" in summary:
        log("census: exists, skip")
        return
    import random  # noqa: PLC0415

    from locma.core import draft as draftmod  # noqa: PLC0415
    from locma.core.engine import make_draft_view  # noqa: PLC0415
    from locma.core.state import GameState, Phase  # noqa: PLC0415
    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
    from locma.policies.ppo import MaskablePPODraftPolicy  # noqa: PLC0415

    cards = load_cards()
    out: dict = {}
    for tag, mk in [("balanced", BalancedDraftPolicy)] + [
        (f"draft_s{s}", (lambda s=s: MaskablePPODraftPolicy(draft_path(s)))) for s in SEEDS
    ]:
        pol = mk()
        items = costs = 0
        curve = [0] * 8
        uniq: set = set()
        for seed in range(CENSUS_N):
            gs = GameState.new(random.Random(seed))
            draftmod.start_draft(gs, cards)
            pols = (pol, BalancedDraftPolicy())  # census seat 0 = the policy under test
            for p in pols:
                p.reset(seed)
            while gs.phase == Phase.DRAFT:
                pick = pols[gs.current].draft_action(make_draft_view(gs), draftmod.draft_legal(gs))
                draftmod.apply_draft_pick(gs, pick)
            deck = gs.picks[0]
            items += sum(1 for c in deck if c.type != 0)
            costs += sum(c.cost for c in deck)
            for c in deck:
                curve[min(c.cost, 7)] += 1
            uniq.update(c.id for c in deck)
        out[tag] = {
            "items_per_deck": round(items / CENSUS_N, 2),
            "mean_deck_cost": round(costs / CENSUS_N / 30, 2),
            "curve": [round(c / CENSUS_N, 1) for c in curve],
            "unique_cards": len(uniq),  # tiny value = degenerate pick policy
        }
    record("census", out)


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


def pilot_gated(arm: str, candidates: list[str], baselines: list[str]) -> dict | None:
    """Pilot -> full for one arm; returns the full verdict or None if gated off."""
    pilot = verdict(f"pilot_{arm}", candidates, baselines, PILOT, start=PRIMARY_START)
    if pilot["mean_delta"] > 0:
        return verdict(f"full_{arm}", candidates, baselines, FULL, start=PRIMARY_START)
    record(f"full_{arm}_skipped", "pilot not positive")
    return None


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E18b learned-draft experiment start ===")

    # Stage A: train (sequential; each run owns the GPU + 16 subprocess envs).
    for s in SEEDS:
        train_seed(s)

    # Stage B: what do the learned decks look like?
    census()

    # Stage C: G1 reactive, own-pilot (the kill gate).
    full_reactive = pilot_gated(
        "reactive",
        [f"ppo:{B0K[s]},{draft_path(s)}" for s in SEEDS],
        [f"ppo:{B0K[s]}" for s in SEEDS],
    )

    # Stage D: G2 planner (independent read; runs regardless of G1).
    full_planner = pilot_gated(
        "planner",
        [ens_spec(s) for s in SEEDS],
        [ENS_ROR],
    )

    # Stage E: G3 fresh-anchor confirms for CI-positive full arms.
    confirms: dict = {}
    for arm, full, cands, bases in (
        (
            "reactive",
            full_reactive,
            [f"ppo:{B0K[s]},{draft_path(s)}" for s in SEEDS],
            [f"ppo:{B0K[s]}" for s in SEEDS],
        ),
        ("planner", full_planner, [ens_spec(s) for s in SEEDS], [ENS_ROR]),
    ):
        if full is not None and full["ci_lo"] > 0:
            confirms[arm] = verdict(f"confirm_{arm}_19M", cands, bases, FULL, start=CONFIRM_START)

    # Pre-registered gates.
    g: dict = {}
    for arm, full in (("reactive", full_reactive), ("planner", full_planner)):
        if full is None:
            g[arm] = {"gate": "pilot-failed"}
            continue
        conf = confirms.get(arm)
        g[arm] = {
            "delta": full["mean_delta"],
            "ci": [full["ci_lo"], full["ci_hi"]],
            "confirmed": bool(conf is not None and conf["ci_lo"] > 0),
        }
        if conf is not None:
            g[arm]["confirm_delta"] = conf["mean_delta"]
            g[arm]["confirm_ci"] = [conf["ci_lo"], conf["ci_hi"]]
    g["draft_seam_open"] = any(isinstance(v, dict) and v.get("confirmed") for v in g.values())
    record("e18b_gates", g)

    log("=== E18b learned-draft experiment DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
