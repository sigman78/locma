"""E33: behavioral autopsy — how does the best REACTIVE policy play differently
from LOOKAHEAD search? (card usage / missed opportunities / turn ordering)

Phase 3 (E32) showed fair search beats the reactive rung 0.788 head-to-head but
only quantified the gap. This maps its SHAPE. The subject (default the reactive
recipe of record, lppo:e29slim) drives full games vs the HARD3 pool; at each of
its turn-starts we shadow-compute, from the identical state:

  * the ORACLE turn plan  — vbeam over the SAME e29slim trio (one turn of pure
    lookahead on the very net the subject uses; any difference is search, not a
    different net). Cheap: one plan per turn.
  * the LETHAL oracle     — lguard.find_lethal, an exhaustive DFS solver (an
    UNBIASED ground truth for "a forced win exists this turn").

Then we compare the subject's ACTUAL turn to the plan on three axes the user
named, plus extras:

  card usage        per-turn Summon / Item / Attack counts, subject vs plan;
                    item-play rate; attacks split face vs trade (tempo vs value).
  missed opportunity missed lethal (lguard says win exists, subject didn't take
                    it); mana left unspent at end of turn; playable cards left in
                    hand unplayed.
  turn ordering     first index where the subject's action sequence diverges
                    from the plan; same_end (the turn commutes to the plan's
                    exact end state — order noise, not a real miss).

Not a promotion gate — a diagnostic. Seeds 12_000_000+ (disjoint from eval/
match/e14 ranges). Raw per-turn rows -> runs/e33-raw.jsonl.gz; aggregates ->
runs/e33-summary.json.

Usage:
    .venv/Scripts/python scripts/e33_reactive_vs_search_behavior.py --smoke
    .venv/Scripts/python scripts/e33_reactive_vs_search_behavior.py --games 15 --workers 6
    .venv/Scripts/python scripts/e33_reactive_vs_search_behavior.py \
        --subject "ppo:depot:e29slim/e29slim_s0.zip|...|...,depot:ldraft/ldraft_s0.zip"
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from locma.harness.parallel import init_eval_worker

E29 = "depot:e29slim/e29slim_s0.zip|depot:e29slim/e29slim_s1.zip|depot:e29slim/e29slim_s2.zip"
LDRAFT = "depot:ldraft/ldraft_s0.zip"
SUBJECT_DEFAULT = f"lppo:{E29},{LDRAFT}"
ORACLE_PATHS = E29.split("|")  # vbeam over the same e29slim trio
OPPONENTS = ("scripted", "max-guard", "max-attack")
SEED0 = 12_000_000
BEAM_WIDTH = 8
MAX_ACTIONS = 20

_CACHE: dict = {}


def _atype(action) -> str:
    from locma.core.actions import Attack, Summon, Use  # noqa: PLC0415

    if isinstance(action, Summon):
        return "S"
    if isinstance(action, Use):
        return "U"
    if isinstance(action, Attack):
        return "A"
    return "P"


def _seq_stats(actions) -> dict:
    """Count action types + face/trade split over an action list (Pass dropped)."""
    c = Counter(_atype(a) for a in actions if _atype(a) != "P")
    atk = [a for a in actions if _atype(a) == "A"]
    return {
        "n": sum(c.values()),
        "S": c.get("S", 0),
        "U": c.get("U", 0),
        "A": c.get("A", 0),
        "A_face": sum(1 for a in atk if a.target_id == -1),
        "A_trade": sum(1 for a in atk if a.target_id != -1),
    }


def _shadow_game(subject_spec: str, opp_spec: str, seed: int, seat: int) -> list[dict]:
    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.actions import Pass  # noqa: PLC0415
    from locma.core.engine import make_battle_view, run_game  # noqa: PLC0415
    from locma.core.state import Phase  # noqa: PLC0415
    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.policies.lguard import find_lethal  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415
    from locma.policies.vbeam import (  # noqa: PLC0415
        EnsembleValueEvaluator,
        _clone_battle,
        plan_turn,
    )

    if "subject" not in _CACHE or _CACHE.get("subject_spec") != subject_spec:
        _CACHE["subject_spec"] = subject_spec
        _CACHE["subject"] = make_policy(subject_spec)
        _CACHE["eval"] = EnsembleValueEvaluator([resolve_path(p) for p in ORACLE_PATHS])
    if opp_spec not in _CACHE:
        _CACHE[opp_spec] = make_policy(opp_spec)
    subject, ev, opp = _CACHE["subject"], _CACHE["eval"], _CACHE[opp_spec]

    turns: list[dict] = []
    st: dict = {"turn": -1, "pending": None, "subj_actions": []}

    def plan_leaf(gs, plan):
        """Simulate the plan; return (end_view_or_None, wins, mana_unspent)."""
        sim = _clone_battle(gs)
        for a in plan:
            if isinstance(a, Pass):
                break
            battlemod.apply_battle(sim, a)
            if sim.phase == Phase.ENDED:
                return None, sim.winner == seat, 0
        return make_battle_view(sim), False, sim.players[seat].mana

    def close(end_view) -> None:
        p = st["pending"]
        if p is None:
            return
        p["subj"] = _seq_stats(st["subj_actions"])
        if end_view is not None:
            me = end_view  # BattleView: my_hand / my_board with .cost, player mana on gs
            p["mana_unspent"] = st.get("end_mana", 0)
            p["playable_unplayed"] = sum(1 for c in me.my_hand if c.cost <= st.get("end_mana", 0))
            p["same_end"] = st.get("plan_end_view") is not None and end_view == st["plan_end_view"]
        turns.append(p)
        st["pending"] = None
        st["subj_actions"] = []

    def hook(s: int, action, gs) -> None:
        if s != seat or gs.phase != Phase.BATTLE:
            return
        if gs.turn != st["turn"]:
            close(None)  # previous turn had no Pass captured (rare)
            st["turn"] = gs.turn
            legal = list(battlemod.battle_legal(gs))
            plan = (
                plan_turn(gs, ev, width=BEAM_WIDTH, max_actions=MAX_ACTIONS)
                if len(legal) >= 2
                else legal
            )
            end_view, plan_wins, plan_mana = plan_leaf(gs, plan)
            line, exhausted = find_lethal(gs)
            ps = _seq_stats(plan)
            st["plan_end_view"] = end_view
            st["pending"] = {
                "opp": opp_spec,
                "seed": seed,
                "seat": seat,
                "turn": gs.turn,
                "nleg0": len(legal),
                "plan": ps,
                "plan_wins": plan_wins,
                "plan_mana_unspent": plan_mana,
                "lethal_exists": bool(line is not None and exhausted),
                "lethal_unknown": not exhausted,
                "subj_seq": [],
                "plan_seq": [_atype(a) for a in plan if _atype(a) != "P"],
            }
        me = gs.players[seat]
        st["end_mana"] = me.mana  # last-seen mana this turn = unspent at Pass
        if st["pending"] is not None and not isinstance(action, Pass):
            st["subj_actions"].append(action)
            st["pending"]["subj_seq"].append(_atype(action))
        if isinstance(action, Pass):
            close(make_battle_view(gs))

    if seat == 0:
        res = run_game(subject, opp, seed, on_pre_step=hook)
    else:
        res = run_game(opp, subject, seed, on_pre_step=hook)
    close(None)  # game may have ended mid-turn

    won = res.winner == seat
    for t in turns:
        # subject won the game ON this turn iff it is the last turn and subject won
        t["subj_won_turn"] = False
        t["game_win"] = won
    if turns and won and res.winner == seat:
        # mark the final subject turn as the winning one only if game ended in a win
        turns[-1]["subj_won_turn"] = res.turns is not None and won
    return turns


def _first_div(subj_seq, plan_seq) -> int | None:
    """First index where the subject's action-type sequence leaves the plan."""
    for i, pt in enumerate(plan_seq):
        if i >= len(subj_seq) or subj_seq[i] != pt:
            return i
    return None if len(subj_seq) == len(plan_seq) else len(plan_seq)


def aggregate(rows: list[dict]) -> dict:
    dec = [r for r in rows if r["nleg0"] >= 2]  # real decisions only
    n = len(dec)

    def mean(f):
        vals = [f(r) for r in dec if f(r) is not None]
        return round(statistics.fmean(vals), 3) if vals else None

    # card usage: per-turn means, subject vs plan
    usage = {
        k: {
            "subject": mean(lambda r, k=k: r["subj"][k]),
            "plan": mean(lambda r, k=k: r["plan"][k]),
        }
        for k in ("n", "S", "U", "A", "A_face", "A_trade")
    }
    item_turns_subj = sum(1 for r in dec if r["subj"]["U"] > 0)
    item_turns_plan = sum(1 for r in dec if r["plan"]["U"] > 0)

    # missed opportunity
    lethal = [r for r in dec if r["lethal_exists"]]
    missed_lethal = sum(1 for r in lethal if not r["subj_won_turn"])
    # ordering
    divs = [_first_div(r["subj_seq"], r["plan_seq"]) for r in dec]
    root_disagree = sum(1 for d in divs if d == 0)
    same_end = sum(1 for r in dec if r.get("same_end"))

    return {
        "n_turns": n,
        "card_usage_per_turn": usage,
        "item_play_rate": {
            "subject": round(item_turns_subj / n, 3),
            "plan": round(item_turns_plan / n, 3),
        },
        "attack_face_share": {
            "subject": mean(
                lambda r: r["subj"]["A_face"] / r["subj"]["A"] if r["subj"]["A"] else None
            ),
            "plan": mean(
                lambda r: r["plan"]["A_face"] / r["plan"]["A"] if r["plan"]["A"] else None
            ),
        },
        "mana_unspent_per_turn": {
            "subject": mean(lambda r: r.get("mana_unspent")),
            "plan": mean(lambda r: r["plan_mana_unspent"]),
        },
        "playable_unplayed_per_turn": mean(lambda r: r.get("playable_unplayed")),
        "missed_lethal": {
            "turns_with_lethal": len(lethal),
            "missed": missed_lethal,
            "rate": round(missed_lethal / len(lethal), 3) if lethal else None,
        },
        "ordering": {
            "root_disagree_rate": round(root_disagree / n, 3),
            "same_end_rate": round(same_end / n, 3),
            "first_div_hist": dict(Counter(str(d) for d in divs).most_common()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", default=SUBJECT_DEFAULT)
    ap.add_argument("--games", type=int, default=15, help="seeds per opponent (x2 seats)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/e33-summary.json")
    ap.add_argument("--raw", default="runs/e33-raw.jsonl.gz")
    args = ap.parse_args()
    games = 2 if args.smoke else args.games

    specs = [
        (args.subject, opp, SEED0 + i, seat)
        for opp in OPPONENTS
        for i in range(games)
        for seat in (0, 1)
    ]
    print(f"E33 behavioral autopsy — subject={args.subject}")
    print(
        f"  {len(specs)} games ({games}/opp x2 x{len(OPPONENTS)} opps), oracle=vbeam:e29slim-trio"
    )

    all_turns: list[dict] = []
    Path(args.raw).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.raw, "wt", encoding="utf-8") as f:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=init_eval_worker) as ex:
                futs = [ex.submit(_shadow_game, *s) for s in specs]
                for k, fut in enumerate(as_completed(futs)):
                    ts = fut.result()
                    all_turns.extend(ts)
                    for t in ts:
                        f.write(json.dumps(t) + "\n")
                    if (k + 1) % 10 == 0:
                        print(f"  {k + 1}/{len(specs)} games, {len(all_turns)} turns", flush=True)
        else:
            for s in specs:
                ts = _shadow_game(*s)
                all_turns.extend(ts)
                for t in ts:
                    f.write(json.dumps(t) + "\n")

    summary = {"subject": args.subject, "games": len(specs), **aggregate(all_turns)}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}  ({len(all_turns)} turns)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
