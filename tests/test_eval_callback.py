from types import SimpleNamespace

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


def test_eval_fires_once_per_bucket_crossing_not_on_modulus():
    """Regression test for the plain-modulus gate (`num_timesteps % eval_freq
    != 0`), which can never fire when num_timesteps only ever advances in
    increments (n_envs) that don't divide eval_freq -- e.g. n_envs=3 against
    eval_freq=10 never lands on a multiple of 10 at all. Bucket-crossing must
    still fire once per eval_freq boundary crossed."""
    cb = WinRateEvalCallback(eval_freq=10, n_games=1, eval_seed=1_000_000)
    cb.model = SimpleNamespace(logger=SimpleNamespace(dump=lambda ts: None))

    fire_count = 0

    def _stub_evaluate():
        nonlocal fire_count
        fire_count += 1
        return 0.5

    cb._evaluate = _stub_evaluate

    cb.num_timesteps = 0
    for _ in range(10):
        cb.num_timesteps += 3  # never a multiple of eval_freq=10
        cb._on_step()

    # ts sequence: 3,6,9,12,15,18,21,24,27,30 -> buckets 0,0,0,1,1,1,2,2,2,3
    # bucket increases at ts=12, 21, 30: three crossings.
    assert fire_count == 3
    assert cb.last_avg_hard3 == 0.5


def _backstop_only_model_and_callback(**cb_kwargs):
    """Build a model+callback pair where `_on_step`'s periodic eval can NEVER
    fire, so `last_avg_hard3` (and any TensorBoard flush) can only come from
    the `_on_training_end` backstop.

    `n_steps=64` + `total_timesteps=64` means the model runs exactly one
    rollout of 64 env steps (SB3 always collects a full `n_steps` rollout, so
    `num_timesteps` only ever takes values 1..64). `eval_freq` far above 64
    guarantees `num_timesteps % eval_freq` is never 0.
    """
    env = _build_env("random", seed=0, n_envs=1, both_seat=True, obs_mode="flat")
    model = _make_model(env, obs_mode="flat", seed=0, verbose=0, ent_coef=0.02, n_steps=64)
    cb_kwargs.setdefault("eval_freq", 10_000_000)
    cb_kwargs.setdefault("n_games", 1)
    cb_kwargs.setdefault("eval_seed", 1_000_000)
    cb = WinRateEvalCallback(**cb_kwargs)
    return model, cb


def test_backstop_flushes_eval_scalars_to_the_logger():
    """Regression test for the backstop never calling `logger.dump`.

    `logger.record` only *stages* a value; `logger.dump` is what writes it to
    the TensorBoard/CSV/stdout writers and clears the staged dict. If
    `_on_training_end` records `eval/avg_hard3` but never dumps, the key would
    still be sitting, unflushed, in `model.logger.name_to_value` after
    `learn()` returns -- nothing else touches the logger after
    `on_training_end()` in SB3's `learn()` loop. This would fail without the
    `self.logger.dump(...)` call added to `_on_training_end`.
    """
    model, cb = _backstop_only_model_and_callback()
    model.learn(total_timesteps=64, callback=cb)

    assert cb.last_avg_hard3 is not None
    assert "eval/avg_hard3" not in model.logger.name_to_value, (
        "eval/avg_hard3 was recorded but never dumped by the backstop"
    )
