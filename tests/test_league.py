import csv

import pytest

from locma.envs.league import DEFAULT_BASELINES, league_pool_specs, write_league_csv


def test_pool_specs_snapshots_then_baselines():
    specs = league_pool_specs(["a.zip", "b.zip"], ["scripted"])
    assert specs == ["ppo:a.zip", "ppo:b.zip", "scripted"]


def test_pool_specs_default_baselines_and_length():
    specs = league_pool_specs(["a.zip"])
    assert specs == ["ppo:a.zip", *DEFAULT_BASELINES]
    assert len(specs) == 1 + len(DEFAULT_BASELINES)


def test_write_league_csv_roundtrip(tmp_path):
    rows = [
        {"round": 0, "snapshot": "round0.zip", "avg_hard3": 0.601, "n_seeds": 150},
        {"round": 1, "snapshot": "round1.zip", "avg_hard3": 0.632, "n_seeds": 150},
    ]
    p = tmp_path / "sub" / "league.csv"
    write_league_csv(p, rows)
    with open(p, newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert [r["round"] for r in got] == ["0", "1"]
    assert got[1]["snapshot"] == "round1.zip"


def test_build_league_opponent_pool_length():
    pytest.importorskip("stable_baselines3")
    from locma.envs.league import build_league_opponent  # noqa: PLC0415

    opp = build_league_opponent(["a.zip", "b.zip"], ("scripted", "max-guard"), seed=0)
    assert len(opp.pool) == 4
    assert opp.name == "league"


def test_league_env_is_single_token_env():
    pytest.importorskip("stable_baselines3")
    from gymnasium import spaces  # noqa: PLC0415

    from locma.envs.league import _league_env, build_league_opponent  # noqa: PLC0415

    opp = build_league_opponent([], ("scripted",), seed=0)
    env = _league_env(opp, seed=0, obs_mode="token")
    try:
        assert env.num_envs == 1
        assert isinstance(env.observation_space, spaces.Dict)
    finally:
        env.close()


def test_smoke_league_two_rounds(tmp_path):
    pytest.importorskip("stable_baselines3")
    from locma.envs.league import run_league  # noqa: PLC0415
    from locma.envs.training import train_agent  # noqa: PLC0415

    base = str(tmp_path / "round0.zip")
    train_agent(
        "random",
        steps=400,
        out=base,
        seed=0,
        verbose=0,
        both_seat=False,
        obs_mode="token",
    )
    rows = run_league(
        base,
        rounds=2,
        steps_per_round=300,
        out_dir=str(tmp_path / "lg"),
        seed=0,
        eval_seeds=2,
        games_per_seed=1,
        verbose=0,
    )
    assert [r["round"] for r in rows] == [0, 1, 2]
    assert (tmp_path / "lg" / "round1.zip").exists()
    assert (tmp_path / "lg" / "round2.zip").exists()
    assert (tmp_path / "lg" / "league.csv").exists()


def test_selfplay_league_rejects_bad_rounds():
    from typer.testing import CliRunner  # noqa: PLC0415

    from locma.cli.app import app  # noqa: PLC0415

    res = CliRunner().invoke(app, ["selfplay-league", "--base", "x.zip", "--rounds", "0"])
    assert res.exit_code != 0


def test_hard3_eval_help_lists_spec():
    from typer.testing import CliRunner  # noqa: PLC0415

    from locma.cli.app import app  # noqa: PLC0415

    res = CliRunner().invoke(app, ["hard3-eval", "--help"])
    assert res.exit_code == 0
    assert "--spec" in res.output
