import pytest

pytest.importorskip("sb3_contrib")  # ML-only

from locma.envs.training import _build_env, _make_model


def _model(obs_mode, **hp):
    env = _build_env("random", seed=0, n_envs=1, both_seat=True, obs_mode=obs_mode)
    return _make_model(env, obs_mode=obs_mode, seed=0, verbose=0, ent_coef=0.02, **hp)


def test_make_model_threads_all_hyperparameters():
    m = _model(
        "token",
        learning_rate=1e-4,
        target_kl=0.025,
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.9,
        clip_range=0.1,
        vf_coef=1.0,
        max_grad_norm=1.0,
    )
    assert m.n_steps == 1024
    assert m.batch_size == 128
    assert m.n_epochs == 5
    assert m.gamma == 0.995
    assert m.gae_lambda == 0.9
    assert m.vf_coef == 1.0
    assert m.max_grad_norm == 1.0
    assert m.target_kl == 0.025
    # learning_rate and clip_range are stored as SB3 schedules (callables):
    assert abs(m.clip_range(1.0) - 0.1) < 1e-9
    assert abs(m.lr_schedule(1.0) - 1e-4) < 1e-9


def test_token_extractor_kwargs_applied():
    m = _model("token", extractor_kwargs={"d_model": 128, "n_layers": 1})
    ek = m.policy.features_extractor_kwargs
    assert ek["d_model"] == 128 and ek["n_layers"] == 1


def test_flat_defaults_unchanged():
    m = _model("flat")
    assert m.n_steps == 2048 and m.batch_size == 64 and m.n_epochs == 10
    assert m.gamma == 0.99 and m.gae_lambda == 0.95 and m.vf_coef == 0.5
