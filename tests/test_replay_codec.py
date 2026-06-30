# tests/test_replay_codec.py
from __future__ import annotations

import copy

from locma.data.cards_db import load_cards
from locma.harness.replay_codec import (
    apply_delta,
    cardlist_version,
    compact_card,
    compact_state,
    diff_state,
    expand_card,
    expand_state,
)

CARDS = {c.id: c for c in load_cards()}


def _base_hand_card(card_id):
    c = CARDS[card_id]
    return {
        "iid": 7,
        "card_id": card_id,
        "atk": c.attack,
        "def": c.defense,
        "abilities": c.abilities,
    }


def _base_board_card(card_id, *, can_attack=True, has_attacked=False):
    d = _base_hand_card(card_id)
    d["can_attack"] = can_attack
    d["has_attacked"] = has_attacked
    return d


def test_base_hand_card_is_iid_cardid_only():
    ref = compact_card(_base_hand_card(3), board=False)
    assert ref == [7, 3]  # no dev map
    assert expand_card(ref, board=False) == _base_hand_card(3)


def test_buffed_minion_carries_only_deviations():
    card = _base_board_card(3)
    card["atk"] += 2
    card["def"] += 2
    ref = compact_card(card, board=True)
    assert ref[0] == 7 and ref[1] == 3 and ref[2] == {"atk": card["atk"], "def": card["def"]}
    assert expand_card(ref, board=True) == card


def test_keyword_gain_and_ward_consumed_roundtrip():
    card = _base_board_card(3)
    card["abilities"] = "---G--"  # gained Guard vs printed "------"
    assert expand_card(compact_card(card, board=True), board=True) == card
    warded = _base_board_card(3)
    warded["abilities"] = "------"  # Ward (W) consumed → mask back toward base
    assert expand_card(compact_card(warded, board=True), board=True) == warded


def test_board_readiness_default_omitted_else_carried():
    ready = _base_board_card(3)  # can_attack=True, has_attacked=False → default
    assert compact_card(ready, board=True) == [7, 3]
    sick = _base_board_card(3, can_attack=False)
    assert compact_card(sick, board=True)[2] == {"can_attack": False}
    swung = _base_board_card(3, can_attack=False, has_attacked=True)
    assert compact_card(swung, board=True)[2] == {"can_attack": False, "has_attacked": True}
    assert expand_card(compact_card(swung, board=True), board=True) == swung


def test_cardlist_version_is_stable_and_prefixed():
    v = cardlist_version()
    assert v.startswith("cl_") and v == cardlist_version()


def test_unknown_card_id_carries_all_stats_verbatim():
    # a card_id absent from the catalog cannot be catalog-filled → all stats ride
    # in the dev map and round-trip losslessly (no data loss).
    UNKNOWN = 999999
    hand = {"iid": 1, "card_id": UNKNOWN, "atk": 5, "def": 4, "abilities": "B-----"}
    ref = compact_card(hand, board=False)
    assert ref[0] == 1 and ref[1] == UNKNOWN
    assert ref[2] == {"atk": 5, "def": 4, "abilities": "B-----"}
    assert expand_card(ref, board=False) == hand
    board = {**hand, "can_attack": False, "has_attacked": True}
    assert expand_card(compact_card(board, board=True), board=True) == board


def test_deviation_is_per_field_independent():
    # only atk deviates (def == printed) → dev carries atk alone, and vice versa;
    # guards against an atk/def field-name swap in the comparison.
    only_atk = _base_board_card(3)
    only_atk["atk"] = CARDS[3].attack + 3
    assert compact_card(only_atk, board=True)[2] == {"atk": only_atk["atk"]}
    assert expand_card(compact_card(only_atk, board=True), board=True) == only_atk
    only_def = _base_board_card(3)
    only_def["def"] = CARDS[3].defense - 1
    assert compact_card(only_def, board=True)[2] == {"def": only_def["def"]}
    assert expand_card(compact_card(only_def, board=True), board=True) == only_def


def _state(current, p0, p1):
    def player(health, hand, board):
        return {
            "health": health,
            "mana": 1,
            "max_mana": 1,
            "damage_counter": 0,
            "bonus_draw": 0,
            "deck_count": 20,
            "hand": hand,
            "board": board,
        }

    return {"current": current, "players": [player(*p0), player(*p1)]}


def test_state_keyframe_roundtrip():
    s = _state(0, (30, [_base_hand_card(3)], []), (28, [], [_base_board_card(5)]))
    assert expand_state(compact_state(s)) == s


def test_delta_scalar_change_only():
    prev = _state(0, (30, [], []), (30, [], []))
    cur = copy.deepcopy(prev)
    cur["players"][1]["health"] = 25
    cur["players"][1]["bonus_draw"] = 1
    d = diff_state(prev, cur)
    assert d == {"p": [{"seat": 1, "s": {"health": 25, "bonus_draw": 1}}]}
    running = copy.deepcopy(prev)
    apply_delta(running, d)
    assert running == cur


def test_delta_summon_moves_card_hand_to_board_and_buffs():
    prev = _state(0, (30, [_base_hand_card(3)], []), (30, [], []))
    cur = copy.deepcopy(prev)
    summoned = cur["players"][0]["hand"].pop()
    summoned = {**summoned, "can_attack": True, "has_attacked": False, "atk": summoned["atk"] + 1}
    cur["players"][0]["board"] = [summoned]
    d = diff_state(prev, cur)
    running = copy.deepcopy(prev)
    apply_delta(running, d)
    assert running == cur


def test_no_change_yields_empty_delta():
    prev = _state(0, (30, [_base_hand_card(3)], []), (30, [], []))
    assert diff_state(prev, copy.deepcopy(prev)) == {}
    running = copy.deepcopy(prev)
    apply_delta(running, {})
    assert running == prev
