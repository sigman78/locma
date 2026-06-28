from dataclasses import dataclass

import pytest

from locma.core.engine import run_game
from locma.policies.mixed import MixedOpponentPolicy
from locma.policies.registry import make_policy, policy_names


@dataclass
class _DV:
    round: int


class _Stub:
    """A full-policy stub tagged so we can see which delegate is active."""

    def __init__(self, tag):
        self.name = f"stub{tag}"
        self.tag = tag
        self.reset_seeds = []

    def draft_action(self, view, legal):
        return self.tag

    def battle_action(self, view, legal, state=None):
        return ("b", self.tag, state)

    def reset(self, seed=None):
        self.reset_seeds.append(seed)


def _pool(n=3):
    return [_Stub(i) for i in range(n)]


def _episode_tags(p, n_episodes):
    """Drive n_episodes of draft and record the active delegate's tag each one.

    Each episode: a round-0 draft (episode boundary) then a round-1 draft (no
    resample). Returns the tag seen at each episode's first (round-0) pick.
    """
    tags = []
    for _ in range(n_episodes):
        tags.append(p.draft_action(_DV(round=0), [0, 1, 2]))
        p.draft_action(_DV(round=1), [0, 1, 2])  # later in same episode: no resample
    return tags


def test_empty_pool_rejected():
    with pytest.raises(ValueError):
        MixedOpponentPolicy([])


def test_weights_must_match_pool():
    with pytest.raises(ValueError, match="weights"):
        MixedOpponentPolicy(_pool(2), weights=[1.0])


def test_weighted_pool_can_force_a_delegate():
    p = MixedOpponentPolicy(_pool(3), seed=0, weights=[0.0, 0.0, 1.0])
    assert _episode_tags(p, 10) == [2] * 10


def test_resamples_across_episodes_for_variety():
    p = MixedOpponentPolicy(_pool(3), seed=0)
    tags = _episode_tags(p, 30)
    assert len(set(tags)) > 1  # the opponent varies across episodes


def test_stable_within_an_episode():
    p = MixedOpponentPolicy(_pool(3), seed=0)
    first = p.draft_action(_DV(round=0), [0, 1, 2])  # episode start -> resample
    # subsequent picks in the same episode (round only rises) must not resample
    assert p.draft_action(_DV(round=1), [0, 1, 2]) == first
    assert p.draft_action(_DV(round=5), [0, 1, 2]) == first


def test_deterministic_from_seed():
    a = MixedOpponentPolicy(_pool(3), seed=7)
    b = MixedOpponentPolicy(_pool(3), seed=7)
    assert _episode_tags(a, 20) == _episode_tags(b, 20)


def test_battle_action_passes_state_through():
    p = MixedOpponentPolicy(_pool(3), seed=0)
    p.draft_action(_DV(round=0), [0, 1, 2])  # select a delegate
    tag = p._active.tag
    assert p.battle_action(None, [], state="GS") == ("b", tag, "GS")


def test_reset_reseeds_and_resets_pool():
    pool = _pool(3)
    p = MixedOpponentPolicy(pool, seed=1)
    before = _episode_tags(p, 10)
    p.reset(1)
    after = _episode_tags(p, 10)
    assert before == after  # reset(1) reproduces the same delegate sequence
    assert all(s.reset_seeds and s.reset_seeds[-1] == 1 for s in pool)


def test_registry_mixed_preset():
    p = make_policy("mixed")
    assert isinstance(p, MixedOpponentPolicy)
    assert p.name == "mixed"
    assert len(p.pool) == 5  # the five baseline opponents
    assert "mixed" not in policy_names()  # hidden: it's a training opponent
    assert make_policy("mixed:7")._seed == 7  # positional seed param


def test_registry_mixed_plays_a_full_game():
    r = run_game(make_policy("mixed"), make_policy("greedy"), seed=0)
    assert r.winner in (0, 1)
