import random

from locma.core.battle import check_winner, end_turn, start_battle
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards


def _drafted():
    gs = GameState.new(random.Random(1))
    start_draft(gs, load_cards())
    for _ in range(60):
        apply_draft_pick(gs, 0)
    return gs


def test_start_battle_deals_hands_and_mana():
    gs = _drafted()
    start_battle(gs)
    assert len(gs.players[0].hand) == 4
    assert len(gs.players[1].hand) == 5
    assert gs.players[0].max_mana == 1 and gs.players[0].mana == 1


def test_turn_ramps_mana_and_untaps():
    gs = _drafted()
    start_battle(gs)
    end_turn(gs)  # to player 1
    assert gs.current == 1
    assert gs.players[1].max_mana == 1
    end_turn(gs)  # back to player 0, turn 3
    assert gs.players[0].max_mana == 2


def test_check_winner():
    gs = _drafted()
    start_battle(gs)
    gs.players[1].health = 0
    check_winner(gs)
    assert gs.winner == 0 and gs.phase == Phase.ENDED
