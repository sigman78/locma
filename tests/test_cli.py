from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from locma.cli.app import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes so Rich-styled help text can be asserted on."""
    return _ANSI_RE.sub("", text)


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


def test_train_zoo_help_lists_command():
    # --help does not import the ML stack, so this is safe without the [ml] extra.
    assert runner.invoke(app, ["train-zoo", "--help"]).exit_code == 0


def test_train_zoo_rejects_zero_steps():
    # The guard fires before any ML import, so this passes with or without [ml].
    assert runner.invoke(app, ["train-zoo", "--steps-per-opponent", "0"]).exit_code != 0


def test_train_rejects_bad_obs_mode():
    # obs_mode validation fires before any ML import; passes without [ml] extra.
    # The guard raises typer.BadParameter with "obs_mode must be 'flat' or 'token'".
    res = runner.invoke(app, ["train", "--steps", "1", "--obs-mode", "bogus"])
    assert res.exit_code != 0
    assert "obs_mode" in res.output


def test_train_zoo_rejects_bad_obs_mode():
    # obs_mode validation fires before any ML import; passes without [ml] extra.
    # The guard raises typer.BadParameter with "obs_mode must be 'flat' or 'token'".
    res = runner.invoke(app, ["train-zoo", "--obs-mode", "bogus"])
    assert res.exit_code != 0
    assert "obs_mode" in res.output


def test_train_help_includes_learning_rate_and_target_kl():
    # --help must exit 0 and the new flags must appear in the plain-text output.
    # Rich adds ANSI styling; strip it before asserting on option names.
    res = runner.invoke(app, ["train", "--help"])
    assert res.exit_code == 0
    plain = _strip_ansi(res.output)
    assert "--learning-rate" in plain
    assert "--target-kl" in plain


def test_train_zoo_help_includes_learning_rate_and_target_kl():
    res = runner.invoke(app, ["train-zoo", "--help"])
    assert res.exit_code == 0
    plain = _strip_ansi(res.output)
    assert "--learning-rate" in plain
    assert "--target-kl" in plain


def test_record_practicum_rejects_bad_obs_mode():
    # Validation fires before any real recording (guard is above the lazy import).
    # --games is intentionally omitted: validation fires before recording starts.
    res = runner.invoke(app, ["record-practicum", "--obs-mode", "bogus"])
    assert res.exit_code != 0
    assert "obs_mode" in res.output


def test_distill_rejects_bad_obs_mode():
    # Validation fires before any ML import; passes without the [ml] extra.
    res = runner.invoke(app, ["distill", "--obs-mode", "bogus"])
    assert res.exit_code != 0
    assert "obs_mode" in res.output


def test_record_practicum_help_includes_obs_mode():
    res = runner.invoke(app, ["record-practicum", "--help"])
    assert res.exit_code == 0
    assert "--obs-mode" in _strip_ansi(res.output)


def test_distill_help_includes_obs_mode():
    res = runner.invoke(app, ["distill", "--help"])
    assert res.exit_code == 0
    assert "--obs-mode" in _strip_ansi(res.output)
