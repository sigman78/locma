import pytest

pytest.importorskip("sb3_contrib")
pytest.importorskip("optuna")

from typer.testing import CliRunner

from locma.cli.app import app


def test_sweep_cli_smoke(tmp_path):
    db = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    r = CliRunner().invoke(
        app,
        [
            "sweep",
            "--storage",
            db,
            "--study-name",
            "cli",
            "--n-trials",
            "1",
            "--n-envs",
            "1",
            "--total-steps",
            "800",
            "--n-games",
            "2",
            "--tb-root",
            str(tmp_path / "tb"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "best" in r.output.lower()
