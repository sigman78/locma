import random

import pytest

from locma.core import battle as battlemod
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.engine import make_battle_view
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.policies.mcts import MCTSBattlePolicy


def _battle_state(seed=0):
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards())
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
    battlemod.start_battle(gs)
    return gs


def test_mcts_requires_forward_model():
    p = MCTSBattlePolicy(iterations=4, seed=0)
    with pytest.raises(ValueError):
        p.battle_action(None, [], state=None)


def test_mcts_returns_a_legal_action():
    gs = _battle_state(seed=1)
    legal = battlemod.battle_legal(gs)
    view = make_battle_view(gs)
    action = MCTSBattlePolicy(iterations=8, seed=0).battle_action(view, legal, state=gs)
    assert action in legal


def test_mcts_does_not_mutate_real_state():
    gs = _battle_state(seed=2)
    legal = battlemod.battle_legal(gs)
    view = make_battle_view(gs)
    before = (gs.turn, gs.current, gs.players[0].health, gs.players[1].health)
    MCTSBattlePolicy(iterations=8, seed=0).battle_action(view, legal, state=gs)
    after = (gs.turn, gs.current, gs.players[0].health, gs.players[1].health)
    assert before == after


def test_mcts_deterministic_from_seed():
    gs = _battle_state(seed=3)
    legal = battlemod.battle_legal(gs)
    view = make_battle_view(gs)
    a1 = MCTSBattlePolicy(iterations=16, seed=42).battle_action(view, legal, state=gs)
    a2 = MCTSBattlePolicy(iterations=16, seed=42).battle_action(view, legal, state=gs)
    assert a1 == a2


def test_clone_battle_shares_cards_and_isolates_mutations():
    """The fast clone shares immutable Card templates but isolates all mutable
    battle state, so simulating on the clone never touches the real state."""
    from locma.policies.mcts import _clone_battle  # noqa: PLC0415

    gs = _battle_state(seed=5)
    clone = _clone_battle(gs)
    hand0 = gs.players[0].hand
    if hand0:
        assert clone.players[0].hand[0].card is hand0[0].card  # Card shared (not copied)
        assert clone.players[0].hand[0] is not hand0[0]  # CardInstance is a distinct copy
    snap = (
        gs.turn,
        gs.current,
        gs.players[0].health,
        gs.players[1].health,
        len(gs.players[0].hand),
        len(gs.players[0].board),
        len(gs.players[0].deck),
    )
    rng = random.Random(0)
    steps = 0
    while clone.phase == Phase.BATTLE and steps < 80:
        legal = battlemod.battle_legal(clone)
        battlemod.apply_battle(clone, rng.choice(legal))
        steps += 1
    after = (
        gs.turn,
        gs.current,
        gs.players[0].health,
        gs.players[1].health,
        len(gs.players[0].hand),
        len(gs.players[0].board),
        len(gs.players[0].deck),
    )
    assert snap == after  # original state untouched by clone simulation
