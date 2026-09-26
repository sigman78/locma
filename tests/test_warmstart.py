"""Tests for locma.envs.warmstart.warm_start (E42 plumbing, part A2).

Uses a tiny locally-trained model (a few hundred steps) rather than the real
depot:e36/e36_gen7.zip artifact for the functional tests, so the default
suite stays fast (matching tests/test_pointer_head.py's convention: no @slow
marker for a `learn()` budget of a few dozen/hundred steps). One test loads
the real depot artifact to confirm `warm_start` handles it end to end; that
one is marked slow per the repo's "fetches art" convention
(pyproject.toml's `slow` marker doc).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
pytest.importorskip("gymnasium")

from sb3_contrib import MaskablePPO  # noqa: E402

from locma.envs.training import _build_env, _make_model  # noqa: E402
from locma.envs.warmstart import warm_start  # noqa: E402


@pytest.fixture
def old_model_path(tmp_path):
    """A tiny slim+pointer token-fx MaskablePPO model, trained for a handful
    of steps — the "old" checkpoint every warm_start test starts from."""
    env = _build_env("random", seed=0, n_envs=1, both_seat=False, obs_mode="token-fx")
    model = _make_model(
        env,
        obs_mode="token-fx",
        seed=0,
        verbose=0,
        ent_coef=0.0,
        pointer_head=True,
        slim=True,
        device="cpu",
    )
    model.learn(total_timesteps=192)
    path = str(tmp_path / "old.zip")
    model.save(path)
    env.close()
    return path


def _logits_and_values(policy, obs):
    policy.set_training_mode(False)
    with torch.no_grad():
        obs_t, _ = policy.obs_to_tensor(obs)
        dist = policy.get_distribution(obs_t)
        values = policy.predict_values(obs_t)
    return dist.distribution.logits, values


def test_warm_start_into_fxh_matches_old_with_hist_dropped(old_model_path):
    """(a) Warm-starting fx -> fxh: action logits and values on the SAME obs
    (hist key dropped) equal the old model's, to 1e-5 — the hist branch is a
    no-op at step 0 regardless of the (real, nonzero) hist input."""
    old = MaskablePPO.load(old_model_path, device="cpu")
    env = _build_env("random", seed=1, n_envs=1, both_seat=False, obs_mode="token-fxh")
    try:
        new = warm_start(old_model_path, env, obs_mode="token-fxh", device="cpu")

        # The hist branch's last Linear must still be exactly zero (E42 contract).
        hist_mlp = new.policy.features_extractor.hist_mlp
        assert hist_mlp is not None
        assert torch.all(hist_mlp[-1].weight == 0)
        assert torch.all(hist_mlp[-1].bias == 0)

        obs = env.reset()
        assert "hist" in obs and (obs["hist"] != 0).any(), "test needs a nonzero hist vector"
        obs_nohist = {k: v for k, v in obs.items() if k != "hist"}

        logits_old, values_old = _logits_and_values(old.policy, obs_nohist)
        logits_new, values_new = _logits_and_values(new.policy, obs)

        assert torch.allclose(logits_old, logits_new, atol=1e-5)
        assert torch.allclose(values_old, values_new, atol=1e-5)
    finally:
        env.close()


def test_warm_start_control_arm_same_obs_mode(old_model_path):
    """(b) Control path: warm-starting fx -> fx reports zero missing keys and
    reproduces the old model's outputs exactly (fresh optimizer aside)."""
    old = MaskablePPO.load(old_model_path, device="cpu")
    env = _build_env("random", seed=2, n_envs=1, both_seat=False, obs_mode="token-fx")
    try:
        new = warm_start(old_model_path, env, obs_mode="token-fx", device="cpu")

        missing, unexpected = new.policy.load_state_dict(old.policy.state_dict(), strict=False)
        assert missing == []
        assert unexpected == []

        obs = env.reset()
        logits_old, values_old = _logits_and_values(old.policy, obs)
        logits_new, values_new = _logits_and_values(new.policy, obs)
        assert torch.allclose(logits_old, logits_new, atol=1e-5)
        assert torch.allclose(values_old, values_new, atol=1e-5)
    finally:
        env.close()


def test_warm_start_fresh_optimizer(old_model_path):
    """(c) The returned model's optimizer carries no old Adam moment state."""
    env = _build_env("random", seed=3, n_envs=1, both_seat=False, obs_mode="token-fxh")
    try:
        new = warm_start(old_model_path, env, obs_mode="token-fxh", device="cpu")
        assert len(new.policy.optimizer.state) == 0
    finally:
        env.close()


def test_warm_start_preserves_hyperparameters(old_model_path):
    """Hyperparameters (including schedule-typed ones, evaluated at the
    start-of-training constant) carry over from the old model."""
    old = MaskablePPO.load(old_model_path, device="cpu")
    env = _build_env("random", seed=4, n_envs=1, both_seat=False, obs_mode="token-fxh")
    try:
        new = warm_start(old_model_path, env, obs_mode="token-fxh", device="cpu")
        old_lr = old.learning_rate(1.0) if callable(old.learning_rate) else old.learning_rate
        old_clip = old.clip_range(1.0) if callable(old.clip_range) else old.clip_range
        new_lr = new.learning_rate(1.0) if callable(new.learning_rate) else new.learning_rate
        new_clip = new.clip_range(1.0) if callable(new.clip_range) else new.clip_range
        assert new_lr == pytest.approx(old_lr)
        assert new_clip == pytest.approx(old_clip)
        assert new.n_steps == old.n_steps
        assert new.batch_size == old.batch_size
        assert new.n_epochs == old.n_epochs
        assert new.gamma == old.gamma
        assert new.gae_lambda == old.gae_lambda
        assert new.max_grad_norm == old.max_grad_norm
        assert new.target_kl == old.target_kl
    finally:
        env.close()


@pytest.mark.slow
def test_warm_start_loads_real_depot_artifact():
    """Warm-starting the real depot:e36/e36_gen7.zip x86-chain endpoint into
    "fxh" works end to end (per the E42 plan: both arms warm from this net)."""
    env = _build_env("random", seed=5, n_envs=1, both_seat=False, obs_mode="token-fxh")
    try:
        new = warm_start("depot:e36/e36_gen7.zip", env, obs_mode="token-fxh", device="cpu")
        assert new.policy.features_extractor.hist_mlp is not None
        obs = env.reset()
        logits, values = _logits_and_values(new.policy, obs)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(values).all()
    finally:
        env.close()
