import random

import pytest

from locma.core import battle as battlemod
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.engine import make_battle_view
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.policies.azlite import AZLiteBattlePolicy


def _battle_state(seed=0):
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards())
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
    battlemod.start_battle(gs)
    return gs


def test_azlite_requires_forward_model():
    p = AZLiteBattlePolicy(iterations=4)
    with pytest.raises(ValueError):
        p.battle_action(None, [], state=None)


def test_azlite_returns_a_legal_action():
    gs = _battle_state(seed=1)
    legal = battlemod.battle_legal(gs)
    view = make_battle_view(gs)
    action = AZLiteBattlePolicy(iterations=16).battle_action(view, legal, state=gs)
    assert action in legal


def test_azlite_single_legal_action_shortcut():
    gs = _battle_state(seed=1)
    view = make_battle_view(gs)
    legal = battlemod.battle_legal(gs)
    only = [legal[0]]
    assert AZLiteBattlePolicy(iterations=64).battle_action(view, only, state=gs) is only[0]


def test_azlite_does_not_mutate_real_state():
    gs = _battle_state(seed=2)
    legal = battlemod.battle_legal(gs)
    view = make_battle_view(gs)
    before = (gs.turn, gs.current, gs.players[0].health, gs.players[1].health)
    AZLiteBattlePolicy(iterations=16).battle_action(view, legal, state=gs)
    after = (gs.turn, gs.current, gs.players[0].health, gs.players[1].health)
    assert before == after


def test_azlite_deterministic():
    # static-value AZ-lite (rollout_turns=0) is a pure function of the state.
    gs = _battle_state(seed=3)
    legal = battlemod.battle_legal(gs)
    view = make_battle_view(gs)
    a1 = AZLiteBattlePolicy(iterations=32).battle_action(view, legal, state=gs)
    a2 = AZLiteBattlePolicy(iterations=32).battle_action(view, legal, state=gs)
    assert a1 == a2


def test_azlite_priors_are_a_distribution():
    gs = _battle_state(seed=4)
    legal = list(battlemod.battle_legal(gs))
    p = AZLiteBattlePolicy(iterations=8)
    priors = p._prior(gs, legal, gs.current)
    assert len(priors) == len(legal)
    assert abs(sum(priors) - 1.0) < 1e-6
    assert all(x >= 0.0 for x in priors)
