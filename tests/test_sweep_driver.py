import pytest

pytest.importorskip("sb3_contrib")
pytest.importorskip("optuna")

from pathlib import Path

from locma.envs.sweep import run_sweep


def test_run_sweep_smoke_and_resumable(tmp_path):
    db = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    # Tiny budget: 1 trial, a few hundred steps, 2 eval games — just exercises the loop.
    s1 = run_sweep(
        storage=db, study_name="smoke", n_trials=1, n_envs=1,
        total_steps=800, eval_freq=400, n_games=2, tb_root=str(tmp_path / "tb"),
    )
    assert len(s1.trials) == 1
    assert s1.trials[0].value is None or 0.0 <= s1.trials[0].value <= 1.0
    # Resumability: a second call on the same storage accumulates, not overwrites.
    s2 = run_sweep(
        storage=db, study_name="smoke", n_trials=1, n_envs=1,
        total_steps=800, eval_freq=400, n_games=2, tb_root=str(tmp_path / "tb"),
    )
    assert len(s2.trials) == 2
    assert Path(tmp_path / "study.db").exists()
