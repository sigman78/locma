"""Tests for locma.envs.selfplay — pure recording helpers (ml-free).

TDD: these tests are written *before* the implementation exists.
"""

import random

import numpy as np

import locma.envs.selfplay as selfplay
from locma.core import battle as battlemod
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.engine import make_battle_view
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.envs.encode import ACTION_SIZE, sem_index


def _battle_state(seed: int = 0) -> GameState:
    """Reusable fixture: draft-resolved battle state (same pattern as test_puct.py)."""
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards())
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
    battlemod.start_battle(gs)
    return gs


# ---------------------------------------------------------------------------
# build_policy_target
# ---------------------------------------------------------------------------


def test_build_policy_target_happy_path():
    """Normal case: ok=True, pi sums to 1, proportions match total, illegal slots = 0."""
    gs = _battle_state(seed=0)
    legal = list(battlemod.battle_legal(gs))
    view = make_battle_view(gs)
    assert len(legal) >= 2, "need ≥2 legal actions"

    total = [3, 1] + [0] * (len(legal) - 2)
    pi, ok = selfplay.build_policy_target(view, legal, total)

    assert ok is True
    assert isinstance(pi, np.ndarray)
    assert pi.shape == (ACTION_SIZE,)
    assert abs(float(pi.sum()) - 1.0) < 1e-6, f"pi.sum()={pi.sum()}"

    # Every slot not mapped by any legal action must be zero
    mapped_idxs = {sem_index(view, a) for a in legal} - {None}
    for j in range(ACTION_SIZE):
        if j not in mapped_idxs:
            assert pi[j] == 0.0, f"slot {j} (not legal) expected 0, got {pi[j]}"

    # Proportions of nonzero-count legal actions must match total[i] / denominator
    valid = [
        (sem_index(view, legal[i]), total[i])
        for i in range(len(legal))
        if sem_index(view, legal[i]) is not None and sem_index(view, legal[i]) < ACTION_SIZE
    ]
    denom = float(sum(cnt for _, cnt in valid))
    for idx, cnt in valid:
        expected = cnt / denom if denom > 0 else 0.0
        assert abs(float(pi[idx]) - expected) < 1e-6, (
            f"slot {idx}: expected {expected}, got {pi[idx]}"
        )


def test_build_policy_target_all_none_drop(monkeypatch):
    """When sem_index always returns None, (zeros, False) is returned — caller drops row."""
    gs = _battle_state(seed=0)
    legal = list(battlemod.battle_legal(gs))
    view = make_battle_view(gs)
    total = [1] * len(legal)

    monkeypatch.setattr(selfplay, "sem_index", lambda v, a: None)

    pi, ok = selfplay.build_policy_target(view, legal, total)

    assert ok is False
    assert isinstance(pi, np.ndarray)
    assert pi.shape == (ACTION_SIZE,)
    assert float(pi.sum()) == 0.0


def test_build_policy_target_all_zero_visits():
    """When every visit count is 0, (pi, False) is returned — the s<=0 guard fires."""
    gs = _battle_state(seed=0)
    legal = list(battlemod.battle_legal(gs))
    view = make_battle_view(gs)
    total = [0] * len(legal)

    pi, ok = selfplay.build_policy_target(view, legal, total)

    assert ok is False
    assert float(pi.sum()) == 0.0


# ---------------------------------------------------------------------------
# outcome_for
# ---------------------------------------------------------------------------


def test_outcome_for_truth_table():
    """Verify all five spec cases."""
    f = selfplay.outcome_for
    assert f(winner=0, seat=0) == +1.0
    assert f(winner=1, seat=0) == -1.0
    assert f(winner=0, seat=1) == -1.0
    assert f(winner=1, seat=1) == +1.0
    assert f(winner=None, seat=0) == 0.0


# ---------------------------------------------------------------------------
# select_move_index
# ---------------------------------------------------------------------------


def test_select_move_index_argmax_past_temp():
    """Past the temp_moves threshold → argmax (index 1 for [1, 5, 2])."""
    rng = random.Random(42)
    idx = selfplay.select_move_index([1, 5, 2], ply=20, temp_moves=10, rng=rng)
    assert idx == 1


def test_select_move_index_sampling_reproducible_and_respects_weights():
    """Sampling branch: deterministic with fixed seed; never picks weight-0 index."""
    total = [0, 1, 0]  # only index 1 has weight

    rng_a = random.Random(7)
    idx_a = selfplay.select_move_index(total, ply=0, temp_moves=10, rng=rng_a)

    rng_b = random.Random(7)
    idx_b = selfplay.select_move_index(total, ply=0, temp_moves=10, rng=rng_b)

    assert idx_a == idx_b, "same seed must produce same index"
    assert idx_a == 1, f"only index 1 has weight; got {idx_a}"
