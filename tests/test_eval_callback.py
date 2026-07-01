import pytest

pytest.importorskip("sb3_contrib")

from locma.envs.eval_callback import WinRateEvalCallback
from locma.envs.training import _build_env, _make_model


def test_callback_logs_avg_hard3_during_short_training():
    env = _build_env("random", seed=0, n_envs=1, both_seat=True, obs_mode="flat")
    model = _make_model(env, obs_mode="flat", seed=0, verbose=0, ent_coef=0.02)
    cb = WinRateEvalCallback(eval_freq=400, n_games=2, eval_seed=1_000_000)
    model.learn(total_timesteps=900, callback=cb)
    assert cb.last_avg_hard3 is not None
    assert 0.0 <= cb.last_avg_hard3 <= 1.0
    # TensorBoard-bound scalar names were recorded at least once:
    assert "eval/avg_hard3" in cb.logged_keys
