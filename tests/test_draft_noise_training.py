import pytest

pytest.importorskip("gymnasium")  # ML-only (the battle env)

import numpy as np

from locma.envs.training import _make_battle_env
from locma.policies.drafts import PartialRandomDraftPolicy


def test_make_battle_env_wraps_opponent_draft_with_noise():
    env = _make_battle_env("greedy", seed=0, draft_noise=3)
    assert env.opponent.name == "greedy+rnd3"
    assert isinstance(env.opponent.draft, PartialRandomDraftPolicy)
    assert env.opponent.draft.k == 3
    # An episode must still play out a full draft + battle handoff.
    obs, _ = env.reset()
    assert obs is not None


def test_make_battle_env_draft_noise_default_off():
    env = _make_battle_env("greedy", seed=0)
    assert not isinstance(env.opponent.draft, PartialRandomDraftPolicy)


def test_make_battle_env_draft_noise_needs_draft_half():
    # `mixed` is a pool policy with no single draft half to wrap.
    with pytest.raises(ValueError, match="draft half"):
        _make_battle_env("mixed", seed=0, draft_noise=3)


def test_draft_noise_changes_training_decks():
    # Same seed, with vs without noise: the first observation (a function of the
    # drafted decks) must differ, i.e. the noise actually reaches the draft.
    a, _ = _make_battle_env("greedy", seed=5).reset(seed=5)
    b, _ = _make_battle_env("greedy", seed=5, draft_noise=6).reset(seed=5)
    assert not np.array_equal(a, b)
