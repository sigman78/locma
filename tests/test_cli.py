from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from locma.cli.app import _disjoint_eval_seeds, app

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


def test_disjoint_eval_seeds_blocks_do_not_overlap():
    # run_match(seed=s, games=games_per_seed) consumes base seeds
    # [s, s + games_per_seed - 1]. Anchors spaced by 1 (the old, buggy behavior)
    # would make consecutive blocks overlap in all but one game; spacing by
    # games_per_seed (the fix) must make every block disjoint from its neighbors.
    seeds, games_per_seed = 40, 25
    anchors = _disjoint_eval_seeds(seeds, games_per_seed)
    assert len(anchors) == seeds
    assert anchors[0] == 1_000_000
    blocks = [set(range(a, a + games_per_seed)) for a in anchors]
    for i in range(len(blocks) - 1):
        assert blocks[i].isdisjoint(blocks[i + 1])
    # Total base seeds played is unchanged from the old `seeds` count: still
    # `seeds * games_per_seed` base seeds overall, just non-overlapping now.
    all_covered = set()
    for b in blocks:
        all_covered |= b
    assert len(all_covered) == seeds * games_per_seed


def test_disjoint_eval_seeds_matches_old_count_when_games_per_seed_is_one():
    # With games_per_seed=1 the old and new spacing coincide (sanity check that the
    # fix is a generalization, not a behavior change for that degenerate case).
    assert _disjoint_eval_seeds(5, 1) == list(range(1_000_000, 1_000_005))


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


# ---------------------------------------------------------------------------
# record-selfplay
# ---------------------------------------------------------------------------


def test_record_selfplay_rejects_zero_games():
    # both self-play and baseline are zero → total < 1
    res = runner.invoke(
        app,
        [
            "record-selfplay",
            "--oracle-path",
            "m.zip",
            "--self-play-games",
            "0",
            "--baseline-games",
            "0",
        ],
    )
    assert res.exit_code != 0


def test_record_selfplay_rejects_negative_self_play_games():
    res = runner.invoke(
        app,
        ["record-selfplay", "--oracle-path", "m.zip", "--self-play-games", "-1"],
    )
    assert res.exit_code != 0


def test_record_selfplay_rejects_zero_k():
    res = runner.invoke(
        app,
        ["record-selfplay", "--oracle-path", "m.zip", "--k", "0"],
    )
    assert res.exit_code != 0


def test_record_selfplay_rejects_zero_i():
    res = runner.invoke(
        app,
        ["record-selfplay", "--oracle-path", "m.zip", "--i", "0"],
    )
    assert res.exit_code != 0


def test_record_selfplay_dispatches(monkeypatch):
    captured = {}

    def fake_record_selfplay(**kwargs):
        captured.update(kwargs)
        return {"n_examples": 42, "failed_games": 0}

    monkeypatch.setattr("locma.envs.selfplay.record_selfplay", fake_record_selfplay)

    res = runner.invoke(
        app,
        [
            "record-selfplay",
            "--oracle-path",
            "oracle.zip",
            "--out",
            "out.npz",
            "--self-play-games",
            "10",
            "--baseline-games",
            "5",
            "--k",
            "3",
            "--i",
            "20",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["oracle_path"] == "oracle.zip"
    assert captured["self_play_games"] == 10
    assert captured["K"] == 3
    assert captured["I"] == 20
    assert "42" in _strip_ansi(res.output)


# ---------------------------------------------------------------------------
# az-train
# ---------------------------------------------------------------------------


def test_az_train_rejects_zero_epochs():
    res = runner.invoke(
        app,
        ["az-train", "--data", "x.npz", "--warm-start", "w.zip", "--epochs", "0"],
    )
    assert res.exit_code != 0


def test_az_train_rejects_bad_val_frac():
    res = runner.invoke(
        app,
        ["az-train", "--data", "x.npz", "--warm-start", "w.zip", "--val-frac", "1.5"],
    )
    assert res.exit_code != 0


def test_az_train_rejects_empty_data():
    # --data not provided at all → validation rejects before import
    res = runner.invoke(
        app,
        ["az-train", "--warm-start", "w.zip"],
    )
    assert res.exit_code != 0


def test_az_train_rejects_zero_batch():
    res = runner.invoke(
        app,
        ["az-train", "--data", "x.npz", "--warm-start", "w.zip", "--batch", "0"],
    )
    assert res.exit_code != 0


def test_az_train_dispatches(monkeypatch):
    captured = {}

    def fake_az_train(**kwargs):
        captured.update(kwargs)
        return {
            "out": "az.zip",
            "val_policy_ce": 0.1234,
            "val_value_mse": 0.5678,
            "n_train": 100,
            "n_val": 20,
        }

    monkeypatch.setattr("locma.envs.az_train.az_train", fake_az_train)

    res = runner.invoke(
        app,
        [
            "az-train",
            "--data",
            "a.npz",
            "--data",
            "b.npz",
            "--warm-start",
            "w.zip",
            "--epochs",
            "5",
            "--batch",
            "128",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["data"] == ["a.npz", "b.npz"]
    assert captured["warm_start"] == "w.zip"
    assert captured["epochs"] == 5
    assert "0.1234" in _strip_ansi(res.output)


# ---------------------------------------------------------------------------
# az-selfplay
# ---------------------------------------------------------------------------


def test_az_selfplay_rejects_zero_iterations():
    res = runner.invoke(app, ["az-selfplay", "--iterations", "0"])
    assert res.exit_code != 0


def test_az_selfplay_rejects_zero_window():
    res = runner.invoke(app, ["az-selfplay", "--window", "0"])
    assert res.exit_code != 0


def test_az_selfplay_dispatches(monkeypatch):
    captured = {}

    def fake_az_selfplay(**kwargs):
        captured.update(kwargs)
        return {
            "best_net": "runs/az-net-2.zip",
            "best_score": 0.75,
            "final_hard3": 0.70,
            "final_h2h": 0.60,
        }

    monkeypatch.setattr("locma.envs.azloop.az_selfplay", fake_az_selfplay)

    res = runner.invoke(
        app,
        [
            "az-selfplay",
            "--warm-start",
            "w.zip",
            "--prefix",
            "runs/az",
            "--iterations",
            "2",
            "--window",
            "1",
            "--k-gen",
            "4",
            "--i-eval",
            "20",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["warm_start"] == "w.zip"
    assert captured["iterations"] == 2
    assert captured["K_gen"] == 4
    assert captured["I_eval"] == 20
    assert "0.750" in _strip_ansi(res.output)
