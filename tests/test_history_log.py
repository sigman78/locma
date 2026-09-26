"""Tests for the E42 public-history plumbing: PlayerState.played/turn_log
(core/battle.py) and the BattleView fields that expose them (core/engine.py).

See docs/e42-public-history-plan.md, "Fairness" and "Plumbing" sections.
"""

from __future__ import annotations

import dataclasses
import random

from locma.core.actions import Pass, Summon, Use
from locma.core.battle import apply_battle, battle_legal, start_battle
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.engine import make_battle_view
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards


def _drafted(seed: int = 1) -> GameState:
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards())
    for _ in range(60):
        apply_draft_pick(gs, 0)
    return gs


def _card_id_in_hand(p, iid: int) -> int:
    for c in p.hand:
        if c.instance_id == iid:
            return c.card.id
    raise AssertionError(f"instance {iid} not in hand")


# ---------------------------------------------------------------------------
# PlayerState.played / turn_log
# ---------------------------------------------------------------------------


def test_fresh_game_state_has_empty_logs():
    gs = GameState.new(random.Random(0))
    for p in gs.players:
        assert p.played == []
        assert p.turn_log == []


def test_summon_and_use_append_played_in_order():
    """Drive real battle actions, greedily preferring Summon/Use over Pass, and
    check p.played matches an independently-recorded trace exactly, in order,
    for BOTH seats."""
    gs = _drafted()
    start_battle(gs)
    recorded = {0: [], 1: []}
    summon_seen = use_seen = False
    for _ in range(400):
        if gs.phase != Phase.BATTLE:
            break
        p = gs.players[gs.current]
        legal = battle_legal(gs)
        action = next((a for a in legal if isinstance(a, Summon)), None)
        if action is None:
            action = next((a for a in legal if isinstance(a, Use)), None)
        if action is None:
            action = Pass()
        if isinstance(action, Summon):
            recorded[gs.current].append(_card_id_in_hand(p, action.card_instance_id))
            summon_seen = True
        elif isinstance(action, Use):
            recorded[gs.current].append(_card_id_in_hand(p, action.item_instance_id))
            use_seen = True
        apply_battle(gs, action)
        if summon_seen and use_seen and gs.turn > 6:
            break

    assert summon_seen and use_seen, "test setup must exercise both Summon and Use"
    assert gs.players[0].played == recorded[0]
    assert gs.players[1].played == recorded[1]


def test_pass_appends_turn_log_for_passing_player_only():
    gs = _drafted()
    start_battle(gs)
    p = gs.players[gs.current]
    seat = gs.current
    mana_before, hand_before = p.mana, len(p.hand)

    apply_battle(gs, Pass())

    assert gs.players[seat].turn_log == [(mana_before, hand_before)]
    assert gs.players[1 - seat].turn_log == []


def test_start_battle_and_start_turn_do_not_log():
    """start_battle calls start_turn (not end_turn) — no log entries yet."""
    gs = _drafted()
    start_battle(gs)
    assert gs.players[0].turn_log == []
    assert gs.players[1].turn_log == []


# ---------------------------------------------------------------------------
# BattleView plumbing
# ---------------------------------------------------------------------------


def test_view_exposes_deck_counts_and_sorted_deck_cards():
    gs = _drafted()
    start_battle(gs)
    view = make_battle_view(gs)
    me = gs.players[gs.current]
    op = gs.players[gs.opponent(gs.current)]
    assert view.my_deck_count == len(me.deck)
    assert view.op_deck_count == len(op.deck)
    assert view.my_deck_cards == tuple(sorted(c.card.id for c in me.deck))
    assert list(view.my_deck_cards) == sorted(view.my_deck_cards)


def test_view_exposes_op_played_and_op_turn_log():
    gs = _drafted()
    start_battle(gs)
    seat = gs.current
    apply_battle(gs, Pass())  # ends `seat`'s turn; gs.current flips to the opponent
    view = make_battle_view(gs)
    op = gs.players[seat]  # the seat that just passed is now "op" from the new view
    assert view.op_played == tuple(op.played)
    assert view.op_turn_log == tuple(op.turn_log)
    assert view.op_turn_log  # non-empty: the Pass logged one entry


def test_fairness_shuffling_own_deck_does_not_change_the_view():
    """The own-deck feature is an order-destroyed multiset: shuffling
    PlayerState.deck in place must leave make_battle_view(gs) unchanged."""
    gs = _drafted()
    start_battle(gs)
    before = make_battle_view(gs)
    random.shuffle(gs.players[gs.current].deck)
    after = make_battle_view(gs)
    assert before == after


def test_view_never_exposes_opponent_hand_or_deck_order():
    gs = _drafted()
    start_battle(gs)
    view = make_battle_view(gs)
    names = {f.name for f in dataclasses.fields(view)}
    assert "op_hand" not in names
    assert "op_deck_cards" not in names
    assert "op_deck" not in names
