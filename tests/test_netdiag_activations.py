"""Tests for locma.stats.activations — torch/SB3 importorskip-gated.

Uses tiny throwaway gymnasium envs (no engine, no cards DB) so these run in
seconds; layer names/shapes must not hardcode the real 308/155 layout.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
gym = pytest.importorskip("gymnasium")
sb3_contrib = pytest.importorskip("sb3_contrib")

from locma.stats.activations import (  # noqa: E402
    collect_activations,
    n_examples,
    practicum_obs,
    reinit_clone,
)

OBS_D, N_ACT = 12, 5


class _TinyEnv(gym.Env):
    """Minimal Box-obs env supplying spaces for MaskablePPO construction."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_D,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACT)

    def reset(self, *, seed=None, options=None):
        return np.zeros(OBS_D, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_D, dtype=np.float32), 0.0, True, False, {}


@pytest.fixture(scope="module")
def flat_policy():
    model = sb3_contrib.MaskablePPO("MlpPolicy", _TinyEnv(), seed=0, verbose=0)
    return model.policy


def test_collect_flat_layers_and_shapes(flat_policy):
    obs = np.random.default_rng(0).standard_normal((300, OBS_D)).astype(np.float32)
    acts, kinds = collect_activations(flat_policy, obs, batch_size=128)
    assert set(acts) == {"pi_a1", "pi_a2", "vf_a1", "vf_a2", "logits"}
    assert acts["pi_a1"].shape == (300, 64)  # SB3 default net_arch
    assert acts["logits"].shape == (300, N_ACT)
    assert kinds["pi_a1"] == "tanh"
    assert kinds["logits"] == "linear"
    # Tanh outputs must be bounded; a Linear walk error would break this.
    assert np.abs(acts["pi_a2"]).max() <= 1.0


def test_collect_batch_size_invariant(flat_policy):
    obs = np.random.default_rng(1).standard_normal((100, OBS_D)).astype(np.float32)
    a1, _ = collect_activations(flat_policy, obs, batch_size=7)
    a2, _ = collect_activations(flat_policy, obs, batch_size=100)
    np.testing.assert_allclose(a1["logits"], a2["logits"], atol=1e-5)


def test_logits_match_policy_distribution(flat_policy):
    """Captured logits must equal the policy's own action distribution logits."""
    obs = np.random.default_rng(2).standard_normal((16, OBS_D)).astype(np.float32)
    acts, _ = collect_activations(flat_policy, obs)
    obs_t, _ = flat_policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = flat_policy.get_distribution(obs_t)
        # Categorical stores NORMALIZED logits (log-probs); normalize ours too.
        ours = torch.log_softmax(torch.as_tensor(acts["logits"]), dim=1)
    np.testing.assert_allclose(ours.numpy(), dist.distribution.logits.cpu().numpy(), atol=1e-5)


def test_reinit_clone_changes_activations_preserves_shapes(flat_policy):
    obs = np.random.default_rng(3).standard_normal((50, OBS_D)).astype(np.float32)
    clone = reinit_clone(flat_policy, seed=123)
    a_trained, _ = collect_activations(flat_policy, obs)
    a_reinit, _ = collect_activations(clone, obs)
    assert a_trained["pi_a1"].shape == a_reinit["pi_a1"].shape
    assert not np.allclose(a_trained["logits"], a_reinit["logits"])
    # The original policy must be untouched by the clone's re-init.
    a_again, _ = collect_activations(flat_policy, obs)
    np.testing.assert_allclose(a_trained["logits"], a_again["logits"], atol=1e-6)


def test_practicum_obs_flat_and_token():
    flat = practicum_obs({"obs": np.ones((4, 6), dtype=np.float64)}, "flat")
    assert flat.dtype == np.float32 and flat.shape == (4, 6)
    tok = practicum_obs(
        {
            "obs_tokens": np.zeros((4, 2, 3)),
            "obs_card_ids": np.zeros((4, 2)),
            "obs_token_mask": np.ones((4, 2)),
            "obs_scalars": np.zeros((4, 5)),
        },
        "token",
    )
    assert set(tok) == {"tokens", "card_ids", "token_mask", "scalars"}
    assert n_examples(tok) == 4
    assert n_examples(flat) == 4
