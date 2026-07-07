"""E16a spell-representation diagnostic: is item underuse a card-OBSERVABILITY problem?

Hypothesis (post-E14a): the flat obs projects every card to (type, cost, atk,
def, abilities) and drops player_hp / enemy_hp / card_draw entirely, so part
of the item pool is under-specified -- four blue items (153 Healing Potion,
154 Poison, 156 Major Life Steal Potion, 160 Minor Life Steal Potion) look
like 0/0 no-ability BLANKS, and 153/154/160 are an exact 3-way observation
alias with near-opposite effects. If representation (not turn-level search,
E14a's H3/H5) drives the 3.3x item underuse, the underuse should concentrate
on hidden-effect items and the net must play the alias trio identically.

Setup (E14a shadow pattern, same net + planner-on-same-net oracle): depot:b0k
s0 plays full games vs the ruler opponents + boardkeep. At the FIRST net
decision of every own battle turn we compute the vbeam whole-turn plan
(width 8, RoR config) from the identical state, then let the net play the
turn out.

Draft override: the ``ppo:`` registry spec pairs the net with
BalancedDraftPolicy, whose _ITEM_DISCOUNT=12 was explicitly tuned to the
net's spell weakness -- natural decks carry ~0-2 items (smoke run: ZERO item
candidates in 16 games), starving the denominator. The net's seat therefore
drafts with RandomDraftPolicy here (~27% items). The planner oracle runs on
the identical states, so net-vs-plan stays matched; but random decks are OOD
for the net, so class contrasts (below), not absolute rates, are the reads.
Known confound for R2: training/eval decks under balanced draft carried only
premium VISIBLE removal, so training exposure correlates with visibility
class. R1 is immune (identical observations force an identical policy
regardless of exposure). Per affordable hand card (cost <= mana at turn start) we record
whether the net actually played it this turn and whether the plan would have.
Same start state, same denominator -> per-card net vs plan use rates are
directly comparable.

Card classes (from cards_db hidden fields php/ehp/draw vs visible payload
atk/def/abilities):
  item_hidden_only  effect entirely invisible in flat obs (153/154/156/160)
  item_mixed        visible payload + hidden rider (e.g. 147 Quick Shot draw)
  item_visible      fully specified by visible features (e.g. 150 Throwing Axe)
  creature_rider    summon effect hidden (29 creatures, e.g. 2 Scuttler)
  creature_plain    vanilla

Pre-registered reads (gates for the v2a obs retrain):
  R1 alias trio: net use rates for 153/154/160 statistically indistinguishable
     (forced: identical observations) while the planner's diverge. Supported =
     planner max-min spread >= 0.10 absolute AND >= 2x the net's spread.
  R2 hidden-effect concentration: underuse ratio U = plan_rate / net_rate.
     Supported = U(item_hidden_only) >= 1.5 x U(item_visible).
  R3 rider items: U(item_mixed) >= 1.25 x U(item_visible) (weaker form).
  R4 creature replication: U(creature_rider) >= 1.25 x U(creature_plain).
Gate: R1 or R2 supported -> Stage B (flat-v2a obs: append php/ehp/draw,
retrain B0 at equal budget) is worth the compute. All reads null -> the
representation hypothesis is dead and the underuse is search/branching
(consistent with E14a H3+H5); do not retrain.

Seed range: 13_000_000+ (1M-8M eval/confirm, 5M match, 9M E14a, 10-12M E15
all disjoint). Raw per-turn records: runs/e16a-raw-<opp>.jsonl.gz. Aggregates
+ verdicts: runs/e16a-summary.json, per-card table runs/e16a-percard.json.
Smoke mode: E16_SMOKE=1 -> tiny grid, separate paths.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
import traceback

SMOKE = os.environ.get("E16_SMOKE") == "1"
WORKERS = 4 if SMOKE else 19
SEEDS_PER_OPP = 2 if SMOKE else 250  # x2 seats = games per opponent
SEED0 = 13_000_000
SUMMARY_PATH = "runs/e16a-smoke.json" if SMOKE else "runs/e16a-summary.json"
PERCARD_PATH = "runs/e16a-smoke-percard.json" if SMOKE else "runs/e16a-percard.json"
LOG_PATH = "runs/e16a-smoke.log" if SMOKE else "runs/e16a.log"
RAW_TMPL = "runs/e16a-smoke-raw-{opp}.jsonl.gz" if SMOKE else "runs/e16a-raw-{opp}.jsonl.gz"

NET_REF = "depot:b0k/b0k_s0.zip"  # reactive recipe of record, single net
BEAM_WIDTH = 8  # the RoR planner config
OPPONENTS = ("scripted", "max-guard", "max-attack", "boardkeep")

ALIAS_TRIO = (153, 154, 160)  # exact flat-obs alias, near-opposite effects

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
# Card classes from the hidden-vs-visible payload split
# ---------------------------------------------------------------------------


def card_classes() -> dict[int, str]:
    from locma.data.cards_db import load_cards  # noqa: PLC0415

    out: dict[int, str] = {}
    for c in load_cards():
        hidden = c.player_hp != 0 or c.enemy_hp != 0 or c.card_draw != 0
        if c.type == 0:  # creature
            out[c.id] = "creature_rider" if hidden else "creature_plain"
        else:
            visible = c.attack != 0 or c.defense != 0 or c.abilities != "------"
            if hidden and not visible:
                out[c.id] = "item_hidden_only"
            elif hidden:
                out[c.id] = "item_mixed"
            else:
                out[c.id] = "item_visible"
    return out


# ---------------------------------------------------------------------------
# Worker: one instrumented game (plan once per own turn, then track the turn)
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _played_iids(actions) -> set[int]:
    from locma.core.actions import Summon, Use  # noqa: PLC0415

    out: set[int] = set()
    for a in actions:
        if isinstance(a, Summon):
            out.add(a.card_instance_id)
        elif isinstance(a, Use):
            out.add(a.item_instance_id)
    return out


def _shadow_game(opp_spec: str, seed: int, seat: int) -> tuple[list, dict]:
    """Play one game (net at `seat`); per own turn, record candidate hand cards
    with net-played vs plan-played flags. Returns (turn_records, game_meta)."""
    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.actions import Pass  # noqa: PLC0415
    from locma.core.engine import make_battle_view, run_game  # noqa: PLC0415
    from locma.core.state import Phase  # noqa: PLC0415
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.policies.drafts import RandomDraftPolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415
    from locma.policies.vbeam import NetValueEvaluator, plan_turn  # noqa: PLC0415

    if "net" not in _CACHE:
        pol = make_policy(f"ppo:{NET_REF}")
        pol.draft = RandomDraftPolicy(seed=0)  # force item exposure (see module doc)
        _CACHE["net"] = pol
        _CACHE["eval"] = NetValueEvaluator(resolve_path(NET_REF))
    if opp_spec not in _CACHE:
        _CACHE[opp_spec] = make_policy(opp_spec)
    net, ev, opp = _CACHE["net"], _CACHE["eval"], _CACHE[opp_spec]

    turns: list[dict] = []
    st: dict = {"turn": -1, "cands": None, "plan_iids": None, "net_iids": None, "meta": None}

    def close_turn(how: str) -> None:
        if st["cands"] is None:
            return
        rec = dict(st["meta"])
        rec["closed"] = how
        rec["cands"] = [
            {**c, "net": int(c["iid"] in st["net_iids"]), "plan": int(c["iid"] in st["plan_iids"])}
            for c in st["cands"]
        ]
        for c in rec["cands"]:
            del c["iid"]  # instance ids are per-game noise; card_id is the key
        turns.append(rec)
        st["cands"] = st["plan_iids"] = st["net_iids"] = st["meta"] = None

    def hook(s: int, action, gs) -> None:
        if s != seat or gs.phase != Phase.BATTLE:
            return
        if gs.turn != st["turn"]:
            close_turn("cap")  # previous turn never saw its Pass (rare)
            st["turn"] = gs.turn
            view = make_battle_view(gs)
            legal = list(battlemod.battle_legal(gs))
            playable_now = _played_iids(legal)
            cands = [
                {
                    "iid": c.instance_id,
                    "cid": c.card_id,
                    "ty": c.type,
                    "cost": c.cost,
                    "leg": int(c.instance_id in playable_now),
                }
                for c in view.my_hand
                if c.cost <= view.me_mana
            ]
            plan = plan_turn(gs, ev, width=BEAM_WIDTH) if len(legal) >= 2 else [legal[0]]
            st["cands"] = cands
            st["plan_iids"] = _played_iids(a for a in plan if not isinstance(a, Pass))
            st["net_iids"] = set()
            st["meta"] = {
                "opp": opp_spec,
                "seed": seed,
                "seat": seat,
                "turn": gs.turn,
                "mana": view.me_mana,
            }
        st["net_iids"] |= _played_iids([action])
        if isinstance(action, Pass):
            close_turn("pass")

    if seat == 0:
        res = run_game(net, opp, seed, on_pre_step=hook)
    else:
        res = run_game(opp, net, seed, on_pre_step=hook)

    close_turn("end")  # a pending turn at game end = the game ended mid-turn
    meta = {"opp": opp_spec, "seed": seed, "seat": seat, "win": res.winner == seat}
    return turns, meta


# ---------------------------------------------------------------------------
# Stage 1: play the instrumented games (per opponent, resumable)
# ---------------------------------------------------------------------------


def run_games(ex, opp: str) -> None:
    tag = f"games_{opp}"
    raw_path = RAW_TMPL.format(opp=opp)
    if tag in summary and os.path.exists(raw_path):
        log(f"{tag}: exists, skip")
        return
    t0 = time.time()
    specs = [(opp, SEED0 + i, seat) for i in range(SEEDS_PER_OPP) for seat in (0, 1)]
    wins = n_turn = n_cand = 0
    with gzip.open(raw_path, "wt", encoding="utf-8") as f:
        results = ex.map(_shadow_game, *zip(*specs, strict=True))
        for k, (turns, meta) in enumerate(results):
            for t in turns:
                f.write(json.dumps(t) + "\n")
                n_cand += len(t["cands"])
            wins += meta["win"]
            n_turn += len(turns)
            if (k + 1) % 50 == 0:
                log(f"{tag}: {k + 1}/{len(specs)} games, {(time.time() - t0) / 60:.1f} min")
    record(
        tag,
        {
            "games": len(specs),
            "wr": round(wins / len(specs), 4),
            "turns": n_turn,
            "candidates": n_cand,
            "minutes": round((time.time() - t0) / 60, 1),
        },
    )


# ---------------------------------------------------------------------------
# Stage 2: aggregate per-card / per-class use rates + pre-registered reads
# ---------------------------------------------------------------------------


def _load_raw() -> list[dict]:
    turns = []
    for opp in OPPONENTS:
        with gzip.open(RAW_TMPL.format(opp=opp), "rt", encoding="utf-8") as f:
            turns.extend(json.loads(line) for line in f)
    return turns


def _rates(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    net = sum(r["net"] for r in rows)
    plan = sum(r["plan"] for r in rows)
    out = {
        "n": n,
        "net_rate": round(net / n, 4),
        "plan_rate": round(plan / n, 4),
    }
    out["underuse"] = round(plan / net, 3) if net else None
    return out


def aggregate() -> None:
    if "reads" in summary:
        log("aggregate: exists, skip")
        return
    classes = card_classes()
    cands = [c for t in _load_raw() for c in t["cands"]]
    log(f"aggregate: {len(cands)} candidate observations")

    by_card: dict = {}
    for c in cands:
        by_card.setdefault(c["cid"], []).append(c)
    percard = {
        str(cid): {"class": classes[cid], **_rates(rows)} for cid, rows in sorted(by_card.items())
    }
    with open(PERCARD_PATH, "w", encoding="utf-8") as f:
        json.dump(percard, f, indent=1)
    log(f"per-card table -> {PERCARD_PATH}")

    by_class: dict = {}
    for c in cands:
        by_class.setdefault(classes[c["cid"]], []).append(c)
    cls = {k: _rates(v) for k, v in sorted(by_class.items())}
    record("by_class", cls)

    # Robustness slice: only candidates with a legal action at turn start
    # (greens with an empty board etc. drop out of the denominator).
    cls_leg = {k: _rates([c for c in v if c["leg"]]) for k, v in sorted(by_class.items())}
    record("by_class_legal_now", cls_leg)

    items = [c for c in cands if classes[c["cid"]].startswith("item")]
    record("items_overall", _rates(items))
    record("trio", {str(cid): percard.get(str(cid), {"n": 0}) for cid in ALIAS_TRIO})

    # ----- pre-registered reads ---------------------------------------------
    reads: dict = {}

    trio = [percard.get(str(cid)) for cid in ALIAS_TRIO]
    if all(t and t["n"] >= 30 for t in trio):
        net_r = [t["net_rate"] for t in trio]
        plan_r = [t["plan_rate"] for t in trio]
        net_spread = max(net_r) - min(net_r)
        plan_spread = max(plan_r) - min(plan_r)
        reads["R1_alias_trio"] = {
            "supported": bool(plan_spread >= 0.10 and plan_spread >= 2 * net_spread),
            "net_spread": round(net_spread, 4),
            "plan_spread": round(plan_spread, 4),
            "min_n": min(t["n"] for t in trio),
        }
    else:
        reads["R1_alias_trio"] = {"supported": None, "reason": "n < 30 for some trio card"}

    def _uratio(a: str, b: str, thresh: float, key: str) -> dict:
        ua, ub = cls.get(a, {}).get("underuse"), cls.get(b, {}).get("underuse")
        ok = ua is not None and ub is not None and ub > 0
        return {
            "supported": bool(ok and ua >= thresh * ub),
            f"U_{a}": ua,
            f"U_{b}": ub,
            "threshold": thresh,
        }

    reads["R2_hidden_concentration"] = _uratio("item_hidden_only", "item_visible", 1.5, "R2")
    reads["R3_rider_items"] = _uratio("item_mixed", "item_visible", 1.25, "R3")
    reads["R4_creature_riders"] = _uratio("creature_rider", "creature_plain", 1.25, "R4")

    gate = bool(
        reads["R1_alias_trio"].get("supported") or reads["R2_hidden_concentration"]["supported"]
    )
    reads["gate_v2a_retrain"] = gate
    record("reads", reads)

    # Top underused items by absolute plan-net gap (context for the writeup).
    gaps = sorted(
        (
            (v["plan_rate"] - v["net_rate"], cid, v)
            for cid, v in percard.items()
            if v["class"].startswith("item") and v["n"] >= 30
        ),
        reverse=True,
    )
    record(
        "top_item_gaps",
        [
            {
                "cid": cid,
                "class": v["class"],
                "n": v["n"],
                "net": v["net_rate"],
                "plan": v["plan_rate"],
            }
            for _, cid, v in gaps[:12]
        ],
    )


def main() -> None:
    os.makedirs("runs", exist_ok=True)
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary.update(json.load(f))
    log(f"=== E16a spell-representation diagnostic start (net={NET_REF}) ===")
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for opp in OPPONENTS:
            run_games(ex, opp)
    aggregate()
    log("=== E16a diagnostic DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("DRIVER CRASHED:\n" + traceback.format_exc())
        sys.exit(1)
