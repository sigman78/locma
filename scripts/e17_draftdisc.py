"""E17 draft item-discount sweep: give the PLANNER the spells it can actually use.

E16a showed the balanced draft's _ITEM_DISCOUNT=12 was tuned to the REACTIVE
net's spell weakness (June 25 sweep: 1.5->6->12 gave 0.47->0.52->0.56), yet the
registry pairs the SAME item-starved draft with the vbeam planner -- which
converts items ~1.7x better per opportunity (net 0.226 vs plan 0.381) and
whose largest per-card gaps are premium removal (Decimate net 0.16 / plan
0.58). The draft-pilot pairing is stale: the deployed planner RoR
(vbeam ensemble, 0.926) plays decks curated for a pilot that cannot use items.

Sweep: item_discount in {3, 0, -2, -4, -8} vs the discount-12 control,
planner side, standard verdict protocol (per-eval-seed avg-hard3, paired
bootstrap CI on common seeds, +0.03 threshold). No training -- eval games
only. Negative discounts are an item BONUS: the zero-discount deck still
carries only ~2.4 items/30 (creature/curve bonuses dominate), so the dose
ladder is d3 1.4 / d0 2.4 / d-2 3.7 / d-4 5.6 / d-8 11.1 items per deck
(balanced-vs-balanced census, N=100).

Stages (idempotent, resumable via runs/e17-summary.json):
  0. census: 200 seeded balanced-vs-balanced drafts per discount -> items/deck
     and mean cost. Arms whose deck composition is within 0.5 items of control
     are dropped (same decks = same games = wasted verdicts).
  1. pilots (10x10, 14M anchors): ens_d{d} = "vbeam:s0|s1|s2,8,20,{d}" vs the
     ENS_ROR control. Gate G1: mean_delta > 0 advances.
  2. full (40x25, 14M): surviving arms vs ENS_ROR. Gate G2: ci_lo > 0.
  3. confirm (40x25, 15M fresh anchors): best G2 arm. Gate G3: ci_lo > 0 ->
     promotion candidate for the planner recipe of record.
  4. reactive guard-rail (10x10, 14M): ppo:b0k_sX,{best_d} vs ppo:b0k_sX.
     Expected null/negative (the 12 tuning was FOR the reactive net); a
     CI-positive here would contradict the June tuning and be flagged.

Seed ranges: eval anchors 14M primary / 15M confirm (1M-13M all used:
eval/confirm/match/E14a/E15/E16a). Smoke: E17_SMOKE=1 -> 2x2 grids,
separate summary/log paths.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E17_SMOKE") == "1"
WORKERS = 19
SUMMARY_PATH = "runs/e17-smoke.json" if SMOKE else "runs/e17-summary.json"
LOG_PATH = "runs/e17-smoke.log" if SMOKE else "runs/e17.log"
PILOT = (2, 2) if SMOKE else (10, 10)  # (eval seeds, games_per_seed)
FULL = (2, 2) if SMOKE else (40, 25)
CENSUS_N = 20 if SMOKE else 200
PRIMARY_START = 14_000_000
CONFIRM_START = 15_000_000

SEEDS = (0, 1, 2)
SHARED = [f"depot:shared/shared_s{s}.zip" for s in SEEDS]
B0K = [f"depot:b0k/b0k_s{s}.zip" for s in SEEDS]
ENS_ROR = "vbeam:" + "|".join(SHARED)  # the 0.926 planner recipe of record
DISCOUNTS = (3.0, 0.0, -2.0, -4.0, -8.0)  # control is the implicit 12.0; <0 = item bonus


def ens_spec(disc: float) -> str:
    return "vbeam:" + "|".join(SHARED) + f",8,20,{disc:g}"


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


# ---------------------------------------------------------------------------
# Stage 0: draft census -- what do the decks actually look like per discount?
# ---------------------------------------------------------------------------


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

    cards = load_cards()
    out: dict = {}
    for disc in (12.0, *DISCOUNTS):
        items = costs = 0
        for seed in range(CENSUS_N):
            gs = GameState.new(random.Random(seed))
            draftmod.start_draft(gs, cards)
            pols = (
                BalancedDraftPolicy(item_discount=disc),
                BalancedDraftPolicy(item_discount=disc),
            )
            while gs.phase == Phase.DRAFT:
                pick = pols[gs.current].draft_action(make_draft_view(gs), draftmod.draft_legal(gs))
                draftmod.apply_draft_pick(gs, pick)
            deck = gs.picks[0]
            items += sum(1 for c in deck if c.type != 0)
            costs += sum(c.cost for c in deck)
        out[f"d{disc:g}"] = {
            "items_per_deck": round(items / CENSUS_N, 2),
            "mean_deck_cost": round(costs / CENSUS_N / 30, 2),
        }
    record("census", out)


def arms_from_census() -> list[float]:
    """Drop discounts whose decks are within 0.5 items of the control's."""
    c = summary["census"]
    base = c["d12"]["items_per_deck"]
    kept = [d for d in DISCOUNTS if abs(c[f"d{d:g}"]["items_per_deck"] - base) >= 0.5]
    dropped = [d for d in DISCOUNTS if d not in kept]
    if dropped:
        log(f"census gate: dropping arms {dropped} (deck composition ~= control)")
    return kept


# ---------------------------------------------------------------------------
# Verdict wrapper (E12 pattern)
# ---------------------------------------------------------------------------


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


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log(f"=== E17 item-discount sweep start (control={ENS_ROR}) ===")

    census()
    arms = arms_from_census()

    # Stage 1: pilots vs the recipe of record (common 14M anchors).
    for d in arms:
        verdict(f"pilot_d{d:g}", [ens_spec(d)], [ENS_ROR], PILOT, start=PRIMARY_START)

    # Gate G1 -> Stage 2 full grids.
    survivors = [d for d in arms if summary[f"pilot_d{d:g}"]["mean_delta"] > 0]
    log(f"G1 pilot gate: {survivors} advance (of {arms})")
    for d in survivors:
        verdict(f"full_d{d:g}", [ens_spec(d)], [ENS_ROR], FULL, start=PRIMARY_START)

    # Gate G2 -> Stage 3 confirm on fresh anchors (best full arm only).
    ci_pos = [d for d in survivors if summary[f"full_d{d:g}"]["ci_lo"] > 0]
    best = max(ci_pos, key=lambda d: summary[f"full_d{d:g}"]["mean_delta"], default=None)
    if best is not None:
        verdict(f"confirm_d{best:g}_15M", [ens_spec(best)], [ENS_ROR], FULL, start=CONFIRM_START)

    # Stage 4: reactive guard-rail with the most item-rich arm that was worth
    # testing (default 3 if census kept it) -- expected null/negative.
    guard_d = best if best is not None else (3.0 if 3.0 in arms else arms[0] if arms else None)
    if guard_d is not None:
        verdict(
            f"guardrail_reactive_d{guard_d:g}",
            [f"ppo:{p},{guard_d:g}" for p in B0K],
            [f"ppo:{p}" for p in B0K],
            PILOT,
            start=PRIMARY_START,
        )

    # Gates summary.
    g: dict = {"arms_after_census": [f"{d:g}" for d in arms], "g1_survivors": [f"{d:g}" for d in survivors]}
    for d in survivors:
        f = summary[f"full_d{d:g}"]
        g[f"full_d{d:g}"] = {"delta": f["mean_delta"], "ci": [f["ci_lo"], f["ci_hi"]], "verdict": f["verdict"]}
    if best is not None:
        c = summary[f"confirm_d{best:g}_15M"]
        g["confirm"] = {
            "arm": f"d{best:g}",
            "delta": c["mean_delta"],
            "ci": [c["ci_lo"], c["ci_hi"]],
            "promote": bool(c["ci_lo"] > 0),
            "spec": ens_spec(best),
        }
    else:
        g["confirm"] = None
    if guard_d is not None:
        r = summary[f"guardrail_reactive_d{guard_d:g}"]
        g["guardrail_flag"] = bool(r["ci_lo"] > 0)  # True would contradict the June tuning
    record("e17_gates", g)

    log("=== E17 item-discount sweep DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
