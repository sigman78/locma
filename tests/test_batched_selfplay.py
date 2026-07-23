"""Tests for the batched-opponent self-play VecEnv (E36)."""

import json

import numpy as np
import pytest

# Top-level import must succeed with no [ml] stack: the SB3 VecEnv import is lazy.
from locma.envs import batched_selfplay


def test_module_imports_without_ml():
    assert hasattr(batched_selfplay, "make_batched_opponent_vecenv")
    assert hasattr(batched_selfplay, "_BatchedOpponentCore")


def _tmp_pool(tmp_path):
    pool = [
        {"spec": "ppo:depot:e29slim/e29slim_s0.zip,depot:ldraft/ldraft_s0.zip", "weight": 2.0},
        {"spec": "scripted", "weight": 1.0},
    ]
    p = tmp_path / "pool.json"
    p.write_text(json.dumps(pool))
    return str(p)


@pytest.mark.slow
def test_vecenv_reset_step_contract(tmp_path):
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from stable_baselines3.common.vec_env import VecEnv  # noqa: PLC0415

    from locma.envs.batched_selfplay import make_batched_opponent_vecenv  # noqa: PLC0415
    from locma.envs.encode import ACTION_SIZE  # noqa: PLC0415

    n = 4
    env = make_batched_opponent_vecenv(_tmp_pool(tmp_path), n_envs=n, seed=1)

    # It must be a real VecEnv so sb3_contrib masking support is detected.
    assert isinstance(env, VecEnv)
    from sb3_contrib.common.maskable.utils import (  # noqa: PLC0415
        get_action_masks,
        is_masking_supported,
    )

    assert is_masking_supported(env)

    obs = env.reset()
    assert set(obs) == {"tokens", "card_ids", "token_mask", "scalars"}
    for v in obs.values():
        assert v.shape[0] == n
        assert np.isfinite(v).all()

    masks = get_action_masks(env)
    assert masks.shape == (n, ACTION_SIZE)
    assert masks.dtype == bool
    assert masks.any(axis=1).all()  # every env has at least one legal action

    # step through enough transitions to force at least one episode boundary
    rng = np.random.default_rng(0)
    saw_done = False
    for _ in range(400):
        masks = get_action_masks(env)
        actions = np.array([int(rng.choice(np.flatnonzero(m))) for m in masks])
        obs, rewards, dones, infos = env.step(actions)
        assert obs["tokens"].shape[0] == n
        assert rewards.shape == (n,) and set(np.unique(rewards)).issubset({-1.0, 0.0, 1.0})
        assert dones.shape == (n,) and dones.dtype == bool
        for i in range(n):
            if dones[i]:
                saw_done = True
                assert rewards[i] in (-1.0, 1.0)  # terminal is win/loss
                assert "terminal_observation" in infos[i]
    assert saw_done, "no episode terminated in 400 steps -- games are not progressing"
    env.close()
