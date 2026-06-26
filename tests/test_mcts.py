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


def test_mcts_defaults_to_heuristic_rollout_and_legacy_still_works():
    assert MCTSBattlePolicy().rollout_turns == 3  # heuristic turn-bounded by default
    gs = _battle_state(seed=7)
    legal = battlemod.battle_legal(gs)
    view = make_battle_view(gs)
    # legacy terminal rollout (rollout_turns=0) still returns a legal action
    a = MCTSBattlePolicy(iterations=8, seed=0, rollout_turns=0).battle_action(view, legal, state=gs)
    assert a in legal


def test_mcts_leaf_value_in_range():
    gs = _battle_state(seed=8)
    v = MCTSBattlePolicy()._leaf_value(gs, 0)
    assert -1.0 <= v <= 1.0


def test_dmcts_requires_state_and_returns_legal():
    from locma.policies.mcts import DMCTSBattlePolicy  # noqa: PLC0415

    with pytest.raises(ValueError):
        DMCTSBattlePolicy().battle_action(None, [], state=None)
    gs = _battle_state(seed=4)
    legal = battlemod.battle_legal(gs)
    view = make_battle_view(gs)
    a = DMCTSBattlePolicy(determinizations=4, iterations=4, seed=0).battle_action(
        view, legal, state=gs
    )
    assert a in legal


def test_dmcts_deterministic_mode_stable_given_obs():
    from locma.policies.mcts import DMCTSBattlePolicy  # noqa: PLC0415

    gs = _battle_state(seed=4)
    legal = battlemod.battle_legal(gs)
    view = make_battle_view(gs)
    p = DMCTSBattlePolicy(determinizations=6, iterations=6, seed=0, deterministic=True)
    assert p.battle_action(view, legal, state=gs) == p.battle_action(view, legal, state=gs)


def test_dmcts_determinize_is_fair_no_state_leak():
    """A fair world uses only public + own-known info: own hand/board kept real,
    own deck CONTENTS preserved but ORDER reshuffled (no future-draw self-leak),
    opponent hand+deck resampled. Real state untouched."""
    from collections import Counter  # noqa: PLC0415

    from locma.policies.mcts import _SAMPLED_ID_BASE, DMCTSBattlePolicy  # noqa: PLC0415

    gs = _battle_state(seed=11)
    me, opp = gs.current, 1 - gs.current
    real_me_hand = [c.instance_id for c in gs.players[me].hand]
    real_me_board = [c.instance_id for c in gs.players[me].board]
    real_me_deck_order = [c.instance_id for c in gs.players[me].deck]
    real_me_deck_contents = Counter(c.card for c in gs.players[me].deck)
    assert len(real_me_deck_order) >= 5  # need a non-trivial deck to see a reshuffle

    det = DMCTSBattlePolicy(seed=123, reshuffle_own=True)._determinize(gs, random.Random(123))
    dm, do = det.players[me], det.players[opp]

    # own hand + board are real (you see them)
    assert [c.instance_id for c in dm.hand] == real_me_hand
    assert [c.instance_id for c in dm.board] == real_me_board
    # own deck: same contents (you know what you drafted) ...
    assert Counter(c.card for c in dm.deck) == real_me_deck_contents
    # ... but reshuffled order — the future-draw sequence is hidden (no self-leak)
    assert [c.instance_id for c in dm.deck] != real_me_deck_order
    # opponent hand + deck resampled (never the real hidden cards)
    assert do.hand and all(c.instance_id >= _SAMPLED_ID_BASE for c in do.hand)
    assert do.deck and all(c.instance_id >= _SAMPLED_ID_BASE for c in do.deck)
    # real state untouched by determinization
    assert [c.instance_id for c in gs.players[me].deck] == real_me_deck_order


def test_dmcts_reshuffle_own_off_reproduces_leak():
    """The opt-out keeps the own deck order real (the old, leaky behaviour) — used
    to A/B how much the self-leak mattered."""
    from locma.policies.mcts import DMCTSBattlePolicy  # noqa: PLC0415

    gs = _battle_state(seed=11)
    me = gs.current
    det = DMCTSBattlePolicy(seed=123, reshuffle_own=False)._determinize(gs, random.Random(123))
    assert [c.instance_id for c in det.players[me].deck] == [
        c.instance_id for c in gs.players[me].deck
    ]
