"""Fidelity tests for subtle LOCM 1.5 rules: the damage-based extra draw (no
runes), deck-out, the second-player bonus mana, and the 50-turn penalty.

Reference: gym-locm Version15BattlePhase and the Strategy-Card-Game-AI
GAME-RULES.md.  There are no runes: for every 5 health a player loses to
opponent damage during a round they draw one extra card next turn.  Deck-out
deals 10 self-damage per missed draw; turn 51+ deals 10 self-damage per turn;
the second player keeps a +1 mana bonus (not counted toward max) until the turn
after fully spending it.
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


def start_turn_for(gs, idx):
    gs.current = idx
    start_turn(gs)


# --- v1.5 draw-on-damage: +1 card per 5 health lost to the opponent ---------


def test_opponent_damage_grants_bonus_draw_per_five():
    gs = _gs()
    gs.players[0].board.append(_attacker(1, 6))
    _resolve_attack(gs, 1, -1)  # 6 face damage from opponent
    p1 = gs.players[1]
    assert p1.health == 24
    assert p1.bonus_draw == 1  # 6 // 5
    assert p1.damage_counter == 1  # remainder carried within the round


def test_damage_accumulates_within_a_round():
    gs = _gs()
    gs.players[0].board.append(_attacker(1, 3))
    gs.players[0].board.append(_attacker(2, 4))
    _resolve_attack(gs, 1, -1)  # counter 3, no draw yet
    assert gs.players[1].bonus_draw == 0
    _resolve_attack(gs, 2, -1)  # counter 7 -> 1 draw, remainder 2
    p1 = gs.players[1]
    assert p1.health == 23
    assert p1.bonus_draw == 1
    assert p1.damage_counter == 2


def test_large_hit_grants_multiple_draws():
    gs = _gs()
    gs.players[0].board.append(_attacker(1, 22))
    _resolve_attack(gs, 1, -1)  # 22 damage -> 4 draws, remainder 2
    p1 = gs.players[1]
    assert p1.health == 8
    assert p1.bonus_draw == 4
    assert p1.damage_counter == 2


def test_self_damage_does_not_grant_draw():
    p = _gs().players[0]
    _change_health(p, 6)  # self / game damage (from_opponent defaults False)
    assert p.health == 24
    assert p.bonus_draw == 0
    assert p.damage_counter == 0


def test_healing_does_not_grant_draw():
    p = _gs().players[0]
    _change_health(p, -10, from_opponent=True)  # negative damage = heal
    assert p.health == 40
    assert p.bonus_draw == 0
    assert p.damage_counter == 0


def test_damage_counter_resets_each_turn():
    gs = _drafted()
    start_battle(gs)
    p1 = gs.players[1]
    p1.damage_counter = 3  # leftover remainder from a prior round
    start_turn_for(gs, 1)
    assert p1.damage_counter == 0  # cleared so fractions never carry across rounds


# --- opening draw + hand cap ------------------------------------------------


def test_first_player_draws_on_first_turn():
    gs = _drafted()
    start_battle(gs)
    # both players reach 5 cards: P0 = 4 opening + 1 first-turn draw, P1 = 5
    assert len(gs.players[0].hand) == 5
    assert len(gs.players[1].hand) == 5


def test_overdraw_leaves_card_in_deck():
    gs = _gs()
    p = gs.players[0]
    p.hand = [_attacker(100 + i, 1) for i in range(8)]  # full hand
    p.deck = [_attacker(200, 1), _attacker(201, 1)]
    draw(gs, 0, 2)
    assert len(p.hand) == 8  # no overdraw
    assert len(p.deck) == 2  # cards stay in deck, not burned


def test_draw_fills_to_cap_then_stops():
    gs = _gs()
    p = gs.players[0]
    p.hand = [_attacker(100 + i, 1) for i in range(7)]
    p.deck = [_attacker(200, 1), _attacker(201, 1), _attacker(202, 1)]
    draw(gs, 0, 3)
    assert len(p.hand) == 8  # drew exactly 1 to reach the cap
    assert len(p.deck) == 2  # only that one card left the deck


# --- deck-out: 10 self-damage per missed draw (no runes) -------------------


def test_deckout_deals_ten_per_missed_draw():
    gs = _gs()
    p = gs.players[0]
    p.deck = []
    draw(gs, 0, 1)
    assert p.health == 20  # 30 - 10
    assert p.bonus_draw == 0  # game damage does not grant draws
    assert p.damage_counter == 0


def test_deckout_repeated_stacks():
    gs = _gs()
    p = gs.players[0]
    p.deck = []
    draw(gs, 0, 1)
    draw(gs, 0, 1)
    assert p.health == 10  # 30 - 20


def test_deckout_can_be_lethal():
    gs = _gs()
    p = gs.players[0]
    p.deck = []
    p.health = 5
    draw(gs, 0, 1)
    assert p.health == -5


def test_deckout_loss_ends_game_at_turn_start():
    gs = _drafted()
    start_battle(gs)
    p1 = gs.players[1]
    p1.deck = []
    p1.health = 5
    end_turn(gs)  # -> player 1's turn starts, draws from empty deck
    assert p1.health == -5
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


# --- 50-turn penalty: 10 self-damage per turn ------------------------------


def test_turn_over_fifty_deals_ten_damage():
    gs = _drafted()
    start_battle(gs)
    gs.turn = 101  # player 0 is now on their 51st turn
    p0 = gs.players[0]
    start_turn_for(gs, 0)
    assert p0.health == 20  # 30 - 10, not a deck-out
    assert p0.deck != []  # deck is NOT emptied (v1.5, unlike v1.2)
