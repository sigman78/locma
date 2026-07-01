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


def _backstop_only_model_and_callback(**cb_kwargs):
    """Build a model+callback pair where `_on_step`'s periodic eval can NEVER
    fire, so `last_avg_hard3` (and any TensorBoard flush / trial report) can
    only come from the `_on_training_end` backstop.

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


def test_backstop_reports_final_eval_to_trial():
    """The backstop must also report to an Optuna trial, mirroring `_on_step`,
    so a trial that never lands on the `eval_freq` modulus still gets a final
    score (Task 5's Optuna driver relies on this)."""
    optuna = pytest.importorskip("optuna")

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    model, cb = _backstop_only_model_and_callback(trial=trial)
    model.learn(total_timesteps=64, callback=cb)

    assert cb.last_avg_hard3 is not None
    reported = study.trials[0].intermediate_values
    assert reported, "backstop should have reported the final eval to the trial"
    assert list(reported.values())[-1] == pytest.approx(cb.last_avg_hard3)


def test_trial_pruning_stops_training():
    """A pruning-enabled trial must raise `optuna.TrialPruned` out of
    `model.learn(...)` when the periodic eval reports a hopeless score."""
    optuna = pytest.importorskip("optuna")

    class _AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    study = optuna.create_study(direction="maximize", pruner=_AlwaysPrune())
    trial = study.ask()

    env = _build_env("random", seed=0, n_envs=1, both_seat=True, obs_mode="flat")
    model = _make_model(env, obs_mode="flat", seed=0, verbose=0, ent_coef=0.02, n_steps=64)
    # eval_freq == n_steps guarantees num_timesteps hits the modulus at the end
    # of the first (and only) rollout, so the periodic path fires at least once.
    cb = WinRateEvalCallback(eval_freq=64, n_games=1, eval_seed=1_000_000, trial=trial)

    with pytest.raises(optuna.TrialPruned):
        model.learn(total_timesteps=200, callback=cb)
