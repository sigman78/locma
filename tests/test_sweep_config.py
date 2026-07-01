import optuna

from locma.envs.sweep import B0_CONFIG, PPOConfig, sample_config, valid


def test_b0_config_matches_spec():
    assert B0_CONFIG.learning_rate == 1e-4
    assert B0_CONFIG.target_kl == 0.025
    assert B0_CONFIG.n_steps == 2048 and B0_CONFIG.batch_size == 64


def test_sample_config_respects_validity_guard():
    # A trial that would pick batch_size > n_steps*n_envs must be flagged invalid.
    bad = PPOConfig(n_steps=1024, batch_size=4096)
    assert not valid(bad, n_envs=1)
    assert valid(PPOConfig(n_steps=2048, batch_size=64), n_envs=1)


def test_sample_config_is_in_range_and_to_kwargs_roundtrips():
    study = optuna.create_study(direction="maximize")
    cfg = sample_config(study.ask())
    assert 3e-5 <= cfg.learning_rate <= 5e-4
    assert cfg.n_steps in (1024, 2048, 4096)
    kw = cfg.to_train_kwargs()
    assert kw["n_steps"] == cfg.n_steps
    assert kw["extractor_kwargs"]["d_model"] == cfg.d_model
