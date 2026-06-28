"""Tests for locma.policies.puct — the shared PUCT core."""

import random

from locma.core import battle as battlemod
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.policies.puct import _mix_root_noise, puct_search


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


# ---------------------------------------------------------------------------
# Root Dirichlet noise tests (P1)
# ---------------------------------------------------------------------------


def test_root_noise_none_is_byte_identical():
    """root_noise=None must produce the exact same visit counts as the no-arg call."""
    gs = _battle_state(seed=0)
    iterations = 50
    rng_seed = 42

    counts_default = puct_search(
        gs, _StubOracle(), iterations=iterations, c_puct=1.5, rng=random.Random(rng_seed)
    )
    counts_none = puct_search(
        gs,
        _StubOracle(),
        iterations=iterations,
        c_puct=1.5,
        rng=random.Random(rng_seed),
        root_noise=None,
    )

    assert counts_default == counts_none, (
        f"root_noise=None changed behaviour: {counts_default} vs {counts_none}"
    )


def test_root_noise_valid_distribution():
    """With root_noise=(0.25, 0.3) the search still runs and counts are valid."""
    gs = _battle_state(seed=0)
    iterations = 50

    counts = puct_search(
        gs,
        _StubOracle(),
        iterations=iterations,
        c_puct=1.5,
        rng=random.Random(7),
        root_noise=(0.25, 0.3),
    )

    assert sum(counts) == iterations, f"visit counts sum to {sum(counts)}, expected {iterations}"
    assert all(c >= 0 for c in counts), "all visit counts must be non-negative"


def test_root_noise_reproducible():
    """Two puct_search calls with the same seed and root_noise produce identical counts."""
    gs = _battle_state(seed=1)
    iterations = 40
    seed = 99
    kwargs = dict(iterations=iterations, c_puct=1.5, root_noise=(0.25, 0.3))

    counts_a = puct_search(gs, _StubOracle(), **kwargs, rng=random.Random(seed))
    counts_b = puct_search(gs, _StubOracle(), **kwargs, rng=random.Random(seed))

    assert counts_a == counts_b, f"Non-reproducible: {counts_a} vs {counts_b}"


def test_mix_root_noise_helper():
    """Unit-test _mix_root_noise directly: eps=0 → P unchanged; eps=1 → pure Dirichlet.

    Also verifies the mixed result always sums ≈1 and all entries ≥ 0.
    """
    P = [0.5, 0.3, 0.2]
    rng = random.Random(17)

    # eps=0: priors unchanged
    mixed_eps0 = _mix_root_noise(P, eps=0.0, alpha=0.3, rng=rng)
    assert mixed_eps0 == P, f"eps=0 should leave priors unchanged, got {mixed_eps0}"

    # eps=1: pure Dirichlet (oracle priors contribute nothing)
    rng2 = random.Random(17)
    # draw the same gamma samples as the implementation would
    g = [rng2.gammavariate(0.3, 1.0) for _ in P]
    s = sum(g) or 1.0
    expected_d = [x / s for x in g]

    rng3 = random.Random(17)
    mixed_eps1 = _mix_root_noise(P, eps=1.0, alpha=0.3, rng=rng3)
    assert mixed_eps1 == expected_d, f"eps=1 should be pure Dirichlet, got {mixed_eps1}"

    # Always sums ≈1 and all non-negative
    rng4 = random.Random(17)
    mixed_mid = _mix_root_noise(P, eps=0.25, alpha=0.3, rng=rng4)
    assert abs(sum(mixed_mid) - 1.0) < 1e-9, f"mixed priors must sum to 1, got {sum(mixed_mid)}"
    assert all(v >= 0.0 for v in mixed_mid), f"mixed priors must be non-negative: {mixed_mid}"
