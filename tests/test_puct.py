"""Tests for locma.policies.puct — the shared PUCT core."""

import random

from locma.core import battle as battlemod
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.policies.puct import puct_search


def _battle_state(seed: int = 0) -> GameState:
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards())
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
    battlemod.start_battle(gs)
    return gs


class _StubOracle:
    """Trivial oracle: uniform priors over actions, fixed value — no net needed."""

    def __init__(self, fixed_value: float = 0.0):
        self._fixed = fixed_value

    def priors(self, sim, actions: list, seat: int) -> list[float]:
        n = len(actions)
        return [1.0 / n] * n

    def value(self, sim, root_seat: int) -> float:
        return self._fixed


def test_puct_visit_counts_sum_to_iterations():
    """Every iteration backprops exactly one root visit; counts sum to iterations."""
    gs = _battle_state(seed=0)  # seed=0 yields 3 legal actions at battle start
    legal = list(battlemod.battle_legal(gs))
    assert len(legal) > 1, "need >1 legal action for this test to be meaningful"

    iterations = 50
    counts = puct_search(gs, _StubOracle(), iterations=iterations, c_puct=1.5, rng=random.Random(0))

    assert len(counts) == len(legal), (
        f"expected {len(legal)} counts (one per legal action), got {len(counts)}"
    )
    assert sum(counts) == iterations, f"visit counts sum to {sum(counts)}, expected {iterations}"
    assert all(c >= 0 for c in counts), "all visit counts must be non-negative"


def test_puct_visit_counts_length_matches_legal_actions():
    """Length of returned counts equals the number of root legal actions."""
    gs = _battle_state(seed=2)
    legal = list(battlemod.battle_legal(gs))
    counts = puct_search(gs, _StubOracle(), iterations=20, c_puct=1.5, rng=random.Random(0))
    assert len(counts) == len(legal)


def test_puct_does_not_mutate_root_state():
    """puct_search must leave the real state untouched."""
    gs = _battle_state(seed=2)
    snap = (gs.turn, gs.current, gs.players[0].health, gs.players[1].health)
    puct_search(gs, _StubOracle(), iterations=30, c_puct=1.5, rng=random.Random(0))
    after = (gs.turn, gs.current, gs.players[0].health, gs.players[1].health)
    assert snap == after


def test_puct_zero_iterations_returns_all_zeros():
    """With 0 iterations, every root visit count is 0 and sum is 0."""
    gs = _battle_state(seed=3)
    counts = puct_search(gs, _StubOracle(), iterations=0, c_puct=1.5, rng=random.Random(0))
    assert sum(counts) == 0
    assert all(c == 0 for c in counts)
