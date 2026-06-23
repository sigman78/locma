from __future__ import annotations

import json

from typer.testing import CliRunner

from locma.cli.app import app

runner = CliRunner()


def test_play_smoke():
    res = runner.invoke(app, ["play", "greedy", "random", "--games", "2", "--seed", "0"])
    assert res.exit_code == 0
    assert "win rate" in res.stdout.lower()


def test_sprt_smoke():
    res = runner.invoke(
        app, ["sprt", "greedy", "--vs", "random", "--max-games", "40", "--batch", "10"]
    )
    assert res.exit_code == 0
    assert "verdict" in res.stdout.lower()


def test_eval_command_is_gone():
    res = runner.invoke(app, ["eval", "greedy"])
    assert res.exit_code != 0


def test_noise_floor_smoke():
    res = runner.invoke(app, ["noise-floor", "random", "--games", "20", "--seed", "0"])
    assert res.exit_code == 0
    assert "resolution limit" in res.stdout.lower()


def test_tournament_matrix_smoke():
    res = runner.invoke(app, ["tournament", "random", "greedy", "--games", "3", "--matrix"])
    assert res.exit_code == 0
    assert "openskill" in res.stdout.lower()


def test_play_log_then_replay_asserts_hash(tmp_path):
    log = tmp_path / "g.jsonl"
    r1 = runner.invoke(
        app,
        ["play", "greedy", "random", "--games", "2", "--seed", "5", "--log", str(log)],
    )
    assert r1.exit_code == 0
    r2 = runner.invoke(app, ["replay", str(log), "--assert-hash"])
    assert r2.exit_code == 0
    assert "ok" in r2.stdout.lower()


def test_replay_detects_tampered_hash(tmp_path):
    log = tmp_path / "g.jsonl"
    runner.invoke(
        app,
        ["play", "greedy", "random", "--games", "1", "--seed", "5", "--log", str(log)],
    )
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    rows[0]["hash"] = "sha256:deadbeef"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = runner.invoke(app, ["replay", str(log), "--assert-hash"])
    assert res.exit_code != 0


def test_play_rejects_zero_games():
    assert runner.invoke(app, ["play", "greedy", "random", "--games", "0"]).exit_code != 0


def test_noise_floor_rejects_zero_games():
    assert runner.invoke(app, ["noise-floor", "random", "--games", "0"]).exit_code != 0


def test_sprt_rejects_zero_max_games():
    assert (
        runner.invoke(app, ["sprt", "greedy", "--vs", "random", "--max-games", "0"]).exit_code != 0
    )


def test_train_help_lists_command():
    # --help does not import the ML stack, so this is safe without the [ml] extra.
    # (rich styles the help text, so assert on exit code, not exact option strings.)
    assert runner.invoke(app, ["train", "--help"]).exit_code == 0


def test_train_rejects_zero_steps():
    # The guard fires before any ML import, so this passes with or without [ml].
    assert runner.invoke(app, ["train", "--steps", "0"]).exit_code != 0
