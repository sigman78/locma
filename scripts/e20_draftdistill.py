"""E20 (quick offline check): can a static per-card priority table, combined
with BalancedDraftPolicy's curve/creature-need context, recover the E18b
learned draft's gain -- without any neural net at draft time?

E18b (learned draft, +0.117 reactive / +0.054 planner, docs/worklog
2026-07-07) trained a draft NET. Its census showed a consistent pattern
across all 3 independently trained seeds: more items (0.76 -> 4-5/30) and a
much cheaper curve (2-drop bucket becomes the biggest, not the 7+ tail). E17
already showed a pure item-discount dial on the scripted BalancedDraftPolicy
can't reach this (monotone negative) -- but that dial only ever moves ONE
number. This asks a sharper question: is the net's policy well approximated
by a CONTEXT-FREE per-card ranking plus the SAME two context terms
BalancedDraftPolicy already tracks (curve-need, creature-deficit), or does it
need something structurally richer (offer-relative reasoning, cross-pick
synergy, etc.)?

Method: log every pick draft_s{0,1,2} (E18b's trained models) makes against a
balanced opponent, then fit a per-card value v[card_id] (one param per card)
plus 2 context weights (w_need, w_creature) via multinomial logistic
regression (3-way softmax -- the draft always offers exactly 3 cards) against
the nets' revealed choices, pooled across all three seeds. Deliverable:
DistilledDraftPolicy (locma/policies/drafts.py), loadable via the registry's
draft-override param as a ``.json`` path (see ``_draft_param``).

Stages (idempotent via runs/e20-summary.json):
  A. log: replay draft_s{0,1,2} vs balanced, EPISODES_PER_SEED episodes/seed,
     recording (offered triplet, curve/creature context, chosen index) for
     every pick the net makes -> runs/e20-picks.npz.
  B. fit: v[card_id] + w_need + w_creature by L-BFGS-B, 80/20 episode-level
     train/held-out split -> runs/e20-fit.json, printed priority list.
  D. heuristic: a SECOND, hand-specified (not fit) candidate -- recalibrate
     BalancedDraftPolicy's curve_target to the 3-seed census average and
     item_discount to whichever grid value reproduces the census item rate
     under that curve -> runs/e20-heuristic.json.
  C/E. quick pilot verdict (10x10, seed 0 only -- a cheap first read, not a
     gated promotion): distilled (B), heuristic (D), and
     [ppo:b0k_s0,e18b_draft_s0.zip] (ceiling), all vs [ppo:b0k_s0] (balanced
     baseline) -- the same G1 matchup E18b used. How much of the +0.117 does
     each recover?

Stage B's per-card fit already came back null (held-out choice accuracy 50%,
win-rate recovery ~0) -- the net isn't a context-free card ranking. Stage D
tests a different, cheaper hypothesis: is the census's own aggregate finding
(curve young + fewer item penalties) sufficient on its own, without fitting
anything to individual picks.

Smoke: E20_SMOKE=1 -> tiny episode counts / pilot grid.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback

import numpy as np

SMOKE = os.environ.get("E20_SMOKE") == "1"
SUMMARY_PATH = "runs/e20-smoke.json" if SMOKE else "runs/e20-summary.json"
LOG_PATH = "runs/e20-smoke.log" if SMOKE else "runs/e20.log"
PICKS_PATH = "runs/e20-smoke-picks.npz" if SMOKE else "runs/e20-picks.npz"
FIT_PATH = "runs/e20-smoke-fit.json" if SMOKE else "runs/e20-fit.json"

SEEDS = (0,) if SMOKE else (0, 1, 2)
DRAFT_TMPL = "runs/e18b_draft_s{s}.zip"
EPISODES_PER_SEED = 6 if SMOKE else 300
L2_VALUES = 0.05  # shrink rarely-offered cards' fitted value toward 0
HELD_OUT_FRAC = 0.2
PILOT = (2, 2) if SMOKE else (10, 10)
PRIMARY_START = 22_000_000  # fresh range, after E19's 20M/21M
PLANNER_START = 23_000_000  # separate fresh range for the planner check
WORKERS = 19

B0K_S0 = "depot:b0k/b0k_s0.zip"
SHARED = [f"depot:shared/shared_s{s}.zip" for s in (0, 1, 2)]
ENS_ROR = "vbeam:" + "|".join(SHARED)  # planner recipe of record, default (balanced) draft

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
    log(f"{key}: {json.dumps(value)[:2000]}")


# ---- Stage A: log revealed picks ---------------------------------------------


def log_picks() -> None:
    if os.path.exists(PICKS_PATH):
        log("log_picks: exists, skip")
        return
    from locma.core import draft as draftmod  # noqa: PLC0415
    from locma.core.engine import make_draft_view  # noqa: PLC0415
    from locma.core.state import GameState, Phase  # noqa: PLC0415
    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
    from locma.policies.ppo import MaskablePPODraftPolicy  # noqa: PLC0415

    cards = load_cards()
    curve_target = BalancedDraftPolicy._CURVE_TARGET
    creature_target = BalancedDraftPolicy._CREATURE_TARGET

    card_ids: list[list[int]] = []
    needs: list[list[float]] = []
    creature_ok: list[list[float]] = []  # 1.0 iff candidate is a creature AND deficit open
    labels: list[int] = []
    episode_id: list[int] = []

    ep = 0
    for s in SEEDS:
        pol = MaskablePPODraftPolicy(DRAFT_TMPL.format(s=s))
        opp = BalancedDraftPolicy()
        for i in range(EPISODES_PER_SEED):
            seed = s * 1_000_000 + i
            agent_seat = i % 2
            gs = GameState.new(random.Random(seed))
            draftmod.start_draft(gs, cards)
            pol.reset(seed)
            opp.reset(seed)
            curve_counts = [0] * 8
            n_creatures = 0
            while gs.phase == Phase.DRAFT:
                view = make_draft_view(gs)
                legal = draftmod.draft_legal(gs)
                if gs.current == agent_seat:
                    row_ids = [cv.card_id for cv in view.offered]
                    row_needs = [
                        max(0, curve_target.get(min(cv.cost, 7), 0) - curve_counts[min(cv.cost, 7)])
                        for cv in view.offered
                    ]
                    row_ok = [
                        1.0 if cv.type == 0 and n_creatures < creature_target else 0.0
                        for cv in view.offered
                    ]
                    pick = pol.draft_action(view, legal)
                    card_ids.append(row_ids)
                    needs.append(row_needs)
                    creature_ok.append(row_ok)
                    labels.append(pick)
                    episode_id.append(ep)
                    chosen = view.offered[pick]
                    curve_counts[min(chosen.cost, 7)] += 1
                    if chosen.type == 0:
                        n_creatures += 1
                else:
                    pick = opp.draft_action(view, legal)
                draftmod.apply_draft_pick(gs, pick)
            ep += 1

    np.savez(
        PICKS_PATH,
        card_ids=np.array(card_ids, dtype=np.int32),
        needs=np.array(needs, dtype=np.float32),
        creature_ok=np.array(creature_ok, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
        episode_id=np.array(episode_id, dtype=np.int64),
    )
    record("log_picks", {"rows": len(labels), "episodes": ep})


# ---- Stage B: fit the multinomial logit --------------------------------------


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit() -> None:
    if "fit" in summary:
        log("fit: exists, skip")
        return
    from scipy.optimize import minimize  # noqa: PLC0415

    from locma.data.cards_db import load_cards  # noqa: PLC0415

    data = np.load(PICKS_PATH)
    card_ids = data["card_ids"]
    needs = data["needs"]
    creature_ok = data["creature_ok"]
    labels = data["labels"]
    episode_id = data["episode_id"]

    n_cards = int(card_ids.max()) + 1
    n = len(labels)

    rng = np.random.default_rng(0)
    episodes = np.unique(episode_id)
    n_held_out = max(1, int(len(episodes) * HELD_OUT_FRAC))
    held_out_eps = set(rng.choice(episodes, size=n_held_out, replace=False).tolist())
    is_held_out = np.array([e in held_out_eps for e in episode_id])
    train_mask = ~is_held_out

    y_onehot = np.zeros((n, 3), dtype=np.float64)
    y_onehot[np.arange(n), labels] = 1.0

    def unpack(theta):
        return theta[:n_cards], theta[n_cards], theta[n_cards + 1]

    def nll_grad(theta):
        v, w_need, w_creature = unpack(theta)
        z = v[card_ids] + w_need * needs + w_creature * creature_ok
        zt, yt = z[train_mask], y_onehot[train_mask]
        p = _softmax(zt)
        loss = -np.sum(yt * np.log(np.clip(p, 1e-12, 1.0))) / zt.shape[0]
        loss += L2_VALUES * np.sum(v * v)

        delta = (p - yt) / zt.shape[0]
        dv = np.zeros(n_cards)
        np.add.at(dv, card_ids[train_mask].ravel(), delta.ravel())
        dv += 2 * L2_VALUES * v
        dw_need = float(np.sum(delta * needs[train_mask]))
        dw_creature = float(np.sum(delta * creature_ok[train_mask]))
        return loss, np.concatenate([dv, [dw_need, dw_creature]])

    theta0 = np.zeros(n_cards + 2)
    res = minimize(nll_grad, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    v, w_need, w_creature = unpack(res.x)

    def acc(mask):
        z = v[card_ids[mask]] + w_need * needs[mask] + w_creature * creature_ok[mask]
        return float(np.mean(z.argmax(axis=1) == labels[mask]))

    names = {c.id: c.name for c in load_cards()}
    order = np.argsort(-v)
    top = [(names.get(int(i), f"#{i}"), round(float(v[i]), 2)) for i in order[:15]]
    bottom = [(names.get(int(i), f"#{i}"), round(float(v[i]), 2)) for i in order[-15:]]

    fit_out = {
        "values": {str(int(i)): round(float(v[i]), 4) for i in range(n_cards) if v[i] != 0},
        "w_need": round(float(w_need), 4),
        "w_creature": round(float(w_creature), 4),
    }
    with open(FIT_PATH, "w", encoding="utf-8") as f:
        json.dump(fit_out, f, indent=2)

    record(
        "fit",
        {
            "rows": n,
            "n_cards_seen": int((np.bincount(card_ids.ravel(), minlength=n_cards) > 0).sum()),
            "train_acc": round(acc(train_mask), 4),
            "held_out_acc": round(acc(is_held_out), 4),
            "w_need": round(float(w_need), 4),
            "w_creature": round(float(w_creature), 4),
            "top15": top,
            "bottom15": bottom,
            "fit_path": FIT_PATH,
        },
    )


# ---- Stage D: census-derived heuristic (hand-specified, not fitted) ---------

# The DISTILLED per-card fit (Stage B) recovered none of E18b's gain despite
# ~50% held-out choice accuracy -- the net isn't using a context-free card
# ranking. This asks a cheaper, more literal question: is E18b's census
# finding (curve shifted young + item discount lowered, see docs/worklog
# 2026-07-07) ENOUGH on its own, hand-specified rather than fit? Recalibrate
# BalancedDraftPolicy's two tuned constants directly from the census: curve
# target = the 3-seed average of the learned decks' curve (rounded to sum 30),
# item_discount = whichever grid value reproduces the census item rate under
# that new curve (E17 already ruled out lowering the discount ALONE under the
# OLD curve; this tests discount + curve TOGETHER, hand-specified).
HEURISTIC_PATH = "runs/e20-smoke-heuristic.json" if SMOKE else "runs/e20-heuristic.json"
# Even discount=0 (no penalty) undershot the census item target under the new
# curve (creature_deficit's own +2.0 bonus still out-competes items while
# n_creatures < 24) -- the grid needs to go NEGATIVE (an item BONUS) to reach it.
DISCOUNT_GRID = (-9.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0)
E18B_SUMMARY_PATH = "runs/e18b-summary.json"


def _sim_items(curve_target: dict[int, int], item_discount: float, n: int) -> float:
    """Mean items/30 for BalancedDraftPolicy(curve_target, item_discount) over n
    seeded solo drafts. Default draft variant: each seat's offers don't depend
    on the other seat's picks, so no real opponent is needed to measure the
    candidate's own resulting deck (see locma/core/draft.py)."""
    from locma.core import draft as draftmod  # noqa: PLC0415
    from locma.core.engine import make_draft_view  # noqa: PLC0415
    from locma.core.state import GameState, Phase  # noqa: PLC0415
    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415

    cards = load_cards()
    pol = BalancedDraftPolicy(item_discount=item_discount, curve_target=curve_target)
    other = BalancedDraftPolicy()
    items = 0
    for seed in range(n):
        gs = GameState.new(random.Random(seed))
        draftmod.start_draft(gs, cards)
        pol.reset(seed)
        other.reset(seed)
        while gs.phase == Phase.DRAFT:
            view = make_draft_view(gs)
            legal = draftmod.draft_legal(gs)
            if gs.current == 0:
                pick = pol.draft_action(view, legal)
            else:
                pick = other.draft_action(view, legal)
            draftmod.apply_draft_pick(gs, pick)
        items += sum(1 for c in gs.picks[0] if c.type != 0)
    return items / n


def heuristic() -> None:
    if "heuristic" in summary:
        log("heuristic: exists, skip")
        return
    with open(E18B_SUMMARY_PATH, encoding="utf-8") as f:
        e18b = json.load(f)
    census = e18b["census"]
    curves = [census[f"draft_s{s}"]["curve"] for s in (0, 1, 2)]
    avg_curve = [sum(c[b] for c in curves) / 3 for b in range(8)]

    target = [round(x) for x in avg_curve]
    diff = 30 - sum(target)
    if diff != 0:
        sign = 1 if diff > 0 else -1
        order = sorted(range(8), key=lambda b: (avg_curve[b] - target[b]) * sign, reverse=True)
        for b in order[: abs(diff)]:
            target[b] += 1 if diff > 0 else -1
    curve_target = {b: target[b] for b in range(8)}
    target_items = sum(census[f"draft_s{s}"]["items_per_deck"] for s in (0, 1, 2)) / 3

    n_sim = 30 if SMOKE else 300
    best = None
    for d in DISCOUNT_GRID:
        items = _sim_items(curve_target, d, n_sim)
        log(f"heuristic calib: discount={d} -> items/30={items:.2f} (target {target_items:.2f})")
        if best is None or abs(items - target_items) < abs(best[1] - target_items):
            best = (d, items)
    item_discount = best[0]

    spec = {"curve_target": curve_target, "item_discount": item_discount}
    with open(HEURISTIC_PATH, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)

    record(
        "heuristic",
        {
            "curve_target": curve_target,
            "balanced_curve_target": {0: 1, 1: 3, 2: 5, 3: 5, 4: 5, 5: 4, 6: 3, 7: 4},
            "item_discount": item_discount,
            "calib_items": round(best[1], 2),
            "target_items": round(target_items, 2),
            "heuristic_path": HEURISTIC_PATH,
        },
    )


# ---- Stage F: elicited card priority (query the net directly, not from real
# drafts) spliced onto the census heuristic's context terms -----------------

# Stage B's fit used REAL in-episode picks, where curve/creature context varies
# together with card identity -- the joint fit confounded the two (w_need came
# out with the wrong sign). Here the context is held FIXED (round 0, empty
# deck) for every query, so there is nothing for card identity to be confounded
# with: each comparison is a clean "which of these 3 cards is better in
# isolation" read. The resulting per-card order is then spliced onto Stage D's
# already-positive census curve_target, using BalancedDraftPolicy's original
# context weights (w_need=3.0, w_creature=2.0) rather than fitting new ones --
# Stage D already showed those two numbers (with the right curve_target) work.
ELICIT_FIT_PATH = "runs/e20-smoke-elicit-fit.json" if SMOKE else "runs/e20-elicit-fit.json"
TRIALS_PER_SEED = 300 if SMOKE else 8000
ELICIT_L2 = 0.02


def _fit_card_values(card_ids: np.ndarray, labels: np.ndarray, n_cards: int, l2: float):
    """Pure per-card multinomial-logit strength (no context features) -- a
    Plackett-Luce fit over 3-way comparisons drawn at a fixed neutral context."""
    from scipy.optimize import minimize  # noqa: PLC0415

    n = len(labels)
    rng = np.random.default_rng(1)
    held_out = rng.random(n) < HELD_OUT_FRAC
    train = ~held_out
    y = np.zeros((n, 3))
    y[np.arange(n), labels] = 1.0

    def nll_grad(v):
        z = v[card_ids]
        zt, yt = z[train], y[train]
        p = _softmax(zt)
        loss = -np.sum(yt * np.log(np.clip(p, 1e-12, 1.0))) / zt.shape[0] + l2 * np.sum(v * v)
        delta = (p - yt) / zt.shape[0]
        dv = np.zeros(n_cards)
        np.add.at(dv, card_ids[train].ravel(), delta.ravel())
        dv += 2 * l2 * v
        return loss, dv

    opts = {"maxiter": 300}
    res = minimize(nll_grad, np.zeros(n_cards), jac=True, method="L-BFGS-B", options=opts)
    v = res.x

    def acc(mask):
        z = v[card_ids[mask]]
        return float(np.mean(z.argmax(axis=1) == labels[mask]))

    return v, acc(train), acc(held_out)


def elicit() -> None:
    if "elicit" in summary:
        log("elicit: exists, skip")
        return
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.core.views import CardView, DraftView  # noqa: PLC0415
    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.envs.encode import draft_action_mask, encode_draft  # noqa: PLC0415

    cards = load_cards()
    by_id = {c.id: c for c in cards}
    ids = np.array(sorted(by_id))
    n_cards = int(ids.max()) + 1
    mask = draft_action_mask([0, 1, 2])

    rng = np.random.default_rng(2)
    all_card_ids: list = []
    all_labels: list = []
    for s in SEEDS:
        model = MaskablePPO.load(DRAFT_TMPL.format(s=s))
        for _ in range(TRIALS_PER_SEED):
            trio = rng.choice(ids, size=3, replace=False)
            offered = tuple(
                CardView(
                    instance_id=-1,
                    card_id=int(cid),
                    type=by_id[int(cid)].type,
                    cost=by_id[int(cid)].cost,
                    attack=by_id[int(cid)].attack,
                    defense=by_id[int(cid)].defense,
                    abilities=by_id[int(cid)].abilities,
                )
                for cid in trio
            )
            view = DraftView(round=0, offered=offered, taken=None)
            obs = encode_draft(view, [])
            idx, _ = model.predict(obs, action_masks=mask, deterministic=True)
            all_card_ids.append(list(trio))
            all_labels.append(int(idx))

    card_ids = np.array(all_card_ids, dtype=np.int32)
    labels = np.array(all_labels, dtype=np.int64)
    v, train_acc, held_out_acc = _fit_card_values(card_ids, labels, n_cards, ELICIT_L2)

    names = {c.id: c.name for c in cards}
    order = np.argsort(-v)
    top = [(names.get(int(i), f"#{i}"), round(float(v[i]), 2)) for i in order[:15]]
    bottom = [(names.get(int(i), f"#{i}"), round(float(v[i]), 2)) for i in order[-15:]]

    curve_target = summary["heuristic"]["curve_target"]
    fit_out = {
        "values": {str(int(i)): round(float(v[i]), 4) for i in range(n_cards) if v[i] != 0},
        "w_need": 3.0,  # BalancedDraftPolicy's original context weights -- not fit here
        "w_creature": 2.0,
        "curve_target": curve_target,
    }
    with open(ELICIT_FIT_PATH, "w", encoding="utf-8") as f:
        json.dump(fit_out, f, indent=2)

    record(
        "elicit",
        {
            "rows": len(labels),
            "train_acc": round(train_acc, 4),
            "held_out_acc": round(held_out_acc, 4),
            "top15": top,
            "bottom15": bottom,
            "elicit_fit_path": ELICIT_FIT_PATH,
        },
    )


# ---- Stage C/E: quick pilot verdict ------------------------------------------


def quick_verdict() -> None:
    from locma.harness.ceiling_eval import _disjoint_eval_seeds, run_verdict  # noqa: PLC0415

    seeds = _disjoint_eval_seeds(*PILOT, start=PRIMARY_START)
    baseline = [f"ppo:{B0K_S0}"]
    arms = {
        "distilled": [f"ppo:{B0K_S0},{FIT_PATH}"],
        "heuristic": [f"ppo:{B0K_S0},{HEURISTIC_PATH}"],
        "elicited": [f"ppo:{B0K_S0},{ELICIT_FIT_PATH}"],
        "ceiling": [f"ppo:{B0K_S0},{DRAFT_TMPL.format(s=0)}"],
    }
    out = dict(summary.get("pilot", {}))
    for tag, cand in arms.items():
        if tag in out:
            log(f"pilot_{tag}: exists, skip")
            continue
        t0 = time.time()
        v = run_verdict(cand, baseline, seeds=seeds, games_per_seed=PILOT[1], workers=WORKERS)
        v = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()}
        v["minutes"] = round((time.time() - t0) / 60, 1)
        out[tag] = v
        log(f"pilot_{tag}: {json.dumps(v)}")
    record("pilot", out)


# ---- Stage G: planner check (elicited heuristic under the 3-critic ensemble) -


def planner_check() -> None:
    if "pilot_planner" in summary:
        log("pilot_planner: exists, skip")
        return
    from locma.harness.ceiling_eval import _disjoint_eval_seeds, run_verdict  # noqa: PLC0415

    seeds = _disjoint_eval_seeds(*PILOT, start=PLANNER_START)
    candidate = [f"{ENS_ROR},8,20,{ELICIT_FIT_PATH}"]
    baseline = [ENS_ROR]
    t0 = time.time()
    v = run_verdict(candidate, baseline, seeds=seeds, games_per_seed=PILOT[1], workers=WORKERS)
    v = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()}
    v["minutes"] = round((time.time() - t0) / 60, 1)
    record("pilot_planner", v)


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log("=== E20 draft-distillation quick check start ===")
    log_picks()
    fit()
    heuristic()
    elicit()
    quick_verdict()
    planner_check()
    log("=== E20 draft-distillation quick check DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
