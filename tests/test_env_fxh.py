"""Tests for BattleEnv obs_mode="token-fxh" (E42 public-history obs variant).

Mirrors tests/test_env_token.py's pattern for "token"/"token-fx".
"""

from __future__ import annotations

import pytest

gym = pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402

from locma.envs.battle_env import BattleEnv  # noqa: E402
from locma.envs.encode import N_HIST  # noqa: E402
from locma.policies.battles import RandomBattlePolicy  # noqa: E402
from locma.policies.composer import Composer  # noqa: E402
from locma.policies.drafts import RandomDraftPolicy  # noqa: E402


def _opp():
    return Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name="opp")


def test_fxh_reset_obs_in_space():
    env = BattleEnv(opponent=_opp(), seed=0, obs_mode="token-fxh")
    obs, _info = env.reset()
    assert isinstance(obs, dict)
    assert set(obs.keys()) == set(env.observation_space.spaces.keys())
    assert "hist" in obs
    assert obs["hist"].shape == (N_HIST,)
    assert obs["hist"].dtype == np.float32
    assert env.observation_space.contains(obs)


def test_fxh_step_a_few_random_legal_actions():
    env = BattleEnv(opponent=_opp(), seed=0, obs_mode="token-fxh")
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = env.action_masks()
        assert mask.any()
        legal_idx = np.flatnonzero(mask)
        idx = int(rng.choice(legal_idx))
        obs, reward, terminated, _truncated, _info = env.step(idx)
        assert reward in (-1.0, 0.0, 1.0)
        assert env.observation_space.contains(obs)
        if terminated:
            env.reset()


def test_fxh_zero_obs_has_hist_key():
    env = BattleEnv(opponent=_opp(), seed=0, obs_mode="token-fxh")
    env.reset()
    zero = env._zero_obs()
    assert "hist" in zero
    assert zero["hist"].shape == (N_HIST,)
    assert np.all(zero["hist"] == 0.0)


def test_fxh_terminal_step_zero_obs_in_space():
    env = BattleEnv(opponent=_opp(), seed=0, obs_mode="token-fxh")
    env.reset()
    MAX_STEPS = 2000
    terminal_obs = None
    for _ in range(MAX_STEPS):
        mask = env.action_masks()
        obs, _reward, terminated, _truncated, _info = env.step(int(np.argmax(mask)))
        if terminated:
            terminal_obs = obs
            break
    assert terminal_obs is not None, "Episode did not terminate within 2000 steps"
    for key, arr in terminal_obs.items():
        assert np.all(arr == 0.0), f"{key}: terminal obs is not all-zero"
    assert env.observation_space.contains(terminal_obs)
