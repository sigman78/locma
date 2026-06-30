# tests/test_replay_codec.py
from __future__ import annotations

from locma.data.cards_db import load_cards
from locma.harness.replay_codec import (
    cardlist_version,
    compact_card,
    expand_card,
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
