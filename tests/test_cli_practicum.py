import json

import numpy as np
from typer.testing import CliRunner

from locma.cli.app import app
from locma.envs.encode import ACTION_SIZE, OBS_SIZE
from locma.envs.practicum import _manifest_path

runner = CliRunner()


def test_record_practicum_cli_writes_dataset(tmp_path):
    out = str(tmp_path / "practicum.npz")
    # cheap teacher so the CLI test stays fast and needs no [ml] extra
    result = runner.invoke(
        app,
        [
            "record-practicum",
            "--teacher",
            "greedy",
            "--opponents",
            "random",
            "--games",
            "2",
            "--out",
            out,
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    data = np.load(out)
    assert data["obs"].shape[1] == OBS_SIZE
    assert data["mask"].shape[1] == ACTION_SIZE
    with open(_manifest_path(out)) as f:
        assert json.load(f)["teacher"] == "greedy"


def test_record_practicum_cli_rejects_bad_teacher(tmp_path):
    result = runner.invoke(
        app, ["record-practicum", "--teacher", "nope", "--out", str(tmp_path / "p.npz")]
    )
    assert result.exit_code != 0


def test_distill_cli_validates_val_frac(tmp_path):
    result = runner.invoke(app, ["distill", "--data", "x.npz", "--val-frac", "1.5"])
    assert result.exit_code != 0
