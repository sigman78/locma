from __future__ import annotations

from typer.testing import CliRunner
from locma.cli.app import app

runner = CliRunner()


def test_play_command_runs():
    r = runner.invoke(app, ["play", "scripted", "random", "--games", "5", "--seed", "0"])
    assert r.exit_code == 0
    assert "win rate" in r.stdout.lower()


def test_eval_command_runs():
    r = runner.invoke(app, ["eval", "greedy", "--vs", "random", "--max-games", "40"])
    assert r.exit_code == 0
    assert "verdict" in r.stdout.lower() or "accept" in r.stdout.lower()
