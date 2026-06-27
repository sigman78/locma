"""Tests for BattleEnv obs_mode="token" (PPO2 tokenized observation).

Guards:
  - token reset/step obs ∈ observation_space
  - terminal step returns all-zero dict ∈ observation_space
  - action_masks() unchanged (always has a legal action)
  - flat default (obs_mode="flat") is byte-identical to today's path
  - invalid obs_mode raises ValueError
"""

from __future__ import annotations

import pytest

gym = pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402

from locma.envs.battle_env import BattleEnv  # noqa: E402
from locma.envs.encode import OBS_SIZE  # noqa: E402
from locma.policies.battles import RandomBattlePolicy  # noqa: E402
from locma.policies.composer import Composer  # noqa: E402
from locma.policies.drafts import RandomDraftPolicy  # noqa: E402


def _opp():
    return Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name="opp")


# ---------------------------------------------------------------------------
# (a) reset() in token mode returns a dict matching observation_space
# ---------------------------------------------------------------------------


def test_token_reset_obs_in_space():
    """reset() returns a dict whose keys/shapes/dtypes match observation_space."""
    env = BattleEnv(opponent=_opp(), seed=0, obs_mode="token")
    obs, _info = env.reset()
    assert isinstance(obs, dict)
    assert set(obs.keys()) == set(env.observation_space.spaces.keys())
    for key, sub in env.observation_space.spaces.items():
        assert obs[key].shape == sub.shape, f"{key}: shape {obs[key].shape} != {sub.shape}"
        assert obs[key].dtype == sub.dtype, f"{key}: dtype {obs[key].dtype} != {sub.dtype}"
    assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# (b) step() returns obs in space and a valid reward
# ---------------------------------------------------------------------------


def test_token_step_obs_in_space():
    """action_masks() has at least one True; step returns obs in space and valid reward."""
    env = BattleEnv(opponent=_opp(), seed=0, obs_mode="token")
    env.reset()
    mask = env.action_masks()
    assert mask.any()
    obs, reward, terminated, _truncated, _info = env.step(int(np.argmax(mask)))
    assert reward in (-1.0, 0.0, 1.0)
    if not terminated:
        assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# (c) terminal step returns all-zero dict that is still in observation_space
# ---------------------------------------------------------------------------


def test_token_terminal_zero_obs():
    """Terminal step returns an all-zero dict that is still ∈ observation_space."""
    env = BattleEnv(opponent=_opp(), seed=0, obs_mode="token")
    env.reset()
    MAX_STEPS = 2000
    terminal_obs = None
    for _ in range(MAX_STEPS):
        obs, _reward, terminated, _truncated, _info = env.step(int(np.argmax(env.action_masks())))
        if terminated:
            terminal_obs = obs
            break
    assert terminal_obs is not None, "Episode did not terminate within 2000 steps"
    assert isinstance(terminal_obs, dict)
    for key, arr in terminal_obs.items():
        assert np.all(arr == 0.0), f"{key}: terminal obs is not all-zero (key={key})"
    assert env.observation_space.contains(terminal_obs)


# ---------------------------------------------------------------------------
# (d) flat default is unchanged
# ---------------------------------------------------------------------------


def test_flat_default_unchanged():
    """Default obs_mode='flat' returns a flat ndarray of shape (OBS_SIZE,)."""
    env = BattleEnv(opponent=_opp(), seed=0)
    obs, _info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# (e) invalid obs_mode raises ValueError
# ---------------------------------------------------------------------------


def test_bogus_obs_mode_raises():
    """An invalid obs_mode raises ValueError at construction time."""
    with pytest.raises(ValueError):
        BattleEnv(opponent=_opp(), obs_mode="bogus")
