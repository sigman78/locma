"""Fidelity tests for subtle LOCM 1.2 rules: runes, deck-out, second-player
bonus mana, and the 50-turn deck-empty rule.

Reference: gym-locm Version12BattlePhase and the Strategy-Card-Game-AI
GAME-RULES.md.  Rune break uses ``health <= next_rune`` and cascades; deck-out
sets HP to the next threshold and breaks that rune; the second player keeps a
+1 mana bonus (not counted toward max) until the turn after fully spending it.
"""

import random

from locma.core.battle import (
    MAX_MANA,
    _change_health,
    _resolve_attack,
    draw,
    end_turn,
    start_battle,
    start_turn,
)
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards


def _gs():
    gs = GameState.new(random.Random(0))
    gs.phase = Phase.BATTLE
    gs.current = 0
    return gs


def _attacker(iid, atk):
    card = Card(1, "X", CardType.CREATURE, 1, atk, 1, normalize_abilities(""), 0, 0, 0)
    inst = CardInstance.from_card(card, iid)
    inst.can_attack = True
    return inst


def _drafted():
    gs = GameState.new(random.Random(1))
    start_draft(gs, load_cards())
    for _ in range(60):
        apply_draft_pick(gs, 0)
    return gs


# --- runes: breaking on threshold crossing ---------------------------------


def test_rune_breaks_at_threshold_grants_bonus_draw():
    gs = _gs()
    gs.players[0].board.append(_attacker(1, 6))
    _resolve_attack(gs, 1, -1)  # 30 -> 24, crosses 25
    p1 = gs.players[1]
    assert p1.health == 24
    assert p1.next_rune == 20
    assert p1.bonus_draw == 1


def test_rune_breaks_exactly_on_threshold():
    gs = _gs()
    gs.players[0].board.append(_attacker(1, 5))
    _resolve_attack(gs, 1, -1)  # 30 -> 25, <= 25 breaks
    p1 = gs.players[1]
    assert p1.health == 25 and p1.next_rune == 20 and p1.bonus_draw == 1


def test_rune_cascade_breaks_multiple_runes():
    gs = _gs()
    gs.players[0].board.append(_attacker(1, 22))
    _resolve_attack(gs, 1, -1)  # 30 -> 8, crosses 25/20/15/10
    p1 = gs.players[1]
    assert p1.health == 8
    assert p1.next_rune == 5
    assert p1.bonus_draw == 4


def test_healing_does_not_restore_broken_runes():
    p = _gs().players[0]
    _change_health(p, 6)  # 30 -> 24, breaks 25
    assert p.next_rune == 20 and p.bonus_draw == 1
    _change_health(p, -20)  # heal to 44
    assert p.health == 44
    assert p.next_rune == 20  # still broken, not restored
    assert p.bonus_draw == 1


# --- deck-out: rune-based, not flat -1 -------------------------------------


def test_deckout_sets_health_to_threshold_and_breaks_rune():
    gs = _gs()
    p = gs.players[0]
    p.deck = []  # empty
    assert p.health == 30 and p.next_rune == 25
    draw(gs, 0, 1)
    assert p.health == 25  # dropped to threshold, not 29
    assert p.next_rune == 20
    assert p.bonus_draw == 1


def test_deckout_repeated_walks_down_runes():
    gs = _gs()
    p = gs.players[0]
    p.deck = []
    draw(gs, 0, 1)  # -> 25 / rune 20
    draw(gs, 0, 1)  # -> 20 / rune 15
    assert p.health == 20 and p.next_rune == 15 and p.bonus_draw == 2


def test_deckout_with_no_runes_left_is_lethal():
    gs = _gs()
    p = gs.players[0]
    p.deck = []
    p.health = 5
    p.next_rune = 0  # all runes already broken
    draw(gs, 0, 1)
    assert p.health == 0


def test_deckout_loss_ends_game_at_turn_start():
    gs = _drafted()
    start_battle(gs)
    p1 = gs.players[1]
    p1.deck = []
    p1.health = 5
    p1.next_rune = 0  # next deck-out is lethal
    end_turn(gs)  # -> player 1's turn starts, draws from empty deck
    assert p1.health == 0
    assert gs.phase == Phase.ENDED
    assert gs.winner == 0


# --- second-player bonus mana ("the coin") ---------------------------------


def test_second_player_starts_with_bonus_mana():
    gs = _drafted()
    start_battle(gs)
    assert gs.players[1].bonus_mana == 1
    assert gs.players[0].bonus_mana == 0


def test_second_player_first_turn_has_extra_mana():
    gs = _drafted()
    start_battle(gs)
    end_turn(gs)  # -> player 1's first turn
    p1 = gs.players[1]
    assert p1.max_mana == 1  # bonus does NOT count toward max
    assert p1.mana == 2  # 1 base + 1 bonus


def start_turn_for(gs, idx):
    gs.current = idx
    start_turn(gs)


def test_bonus_mana_retained_when_not_fully_spent():
    gs = _drafted()
    start_battle(gs)
    p1 = gs.players[1]
    p1.max_mana = 3
    p1.mana = 1  # ended previous turn with leftover mana
    start_turn_for(gs, 1)
    assert p1.bonus_mana == 1
    assert p1.mana == 5  # 4 base (ramped from 3) + 1 bonus


def test_bonus_mana_lost_after_fully_spending():
    gs = _drafted()
    start_battle(gs)
    p1 = gs.players[1]
    p1.max_mana = 3
    p1.mana = 0  # spent everything last turn
    start_turn_for(gs, 1)
    assert p1.bonus_mana == 0
    assert p1.mana == 4  # 4 base, no bonus


def test_bonus_mana_allows_thirteen():
    gs = _drafted()
    start_battle(gs)
    p1 = gs.players[1]
    p1.max_mana = MAX_MANA  # 12
    p1.mana = 5  # not fully spent
    start_turn_for(gs, 1)
    assert p1.max_mana == MAX_MANA
    assert p1.mana == MAX_MANA + 1  # 13


# --- 50-turn deck-empty rule ----------------------------------------------


def test_turn_over_fifty_empties_deck():
    gs = _drafted()
    start_battle(gs)
    gs.turn = 101  # player 0 is now on their 51st turn
    p0 = gs.players[0]
    health_before = p0.health
    start_turn_for(gs, 0)
    assert p0.deck == []
    assert p0.health < health_before  # deck-out triggered
