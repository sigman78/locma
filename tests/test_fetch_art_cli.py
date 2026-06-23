# tests/test_fetch_art_cli.py
from __future__ import annotations

from typer.testing import CliRunner

import locma.data.fetch as fetch
from locma.cli.app import app

runner = CliRunner()


def test_fetch_art_cli_passes_force_and_prints_notice(monkeypatch):
    seen = {}

    def stub(force=False):
        seen["force"] = force
        return 7

    monkeypatch.setattr(fetch, "fetch_art", stub)
    result = runner.invoke(app, ["fetch-art", "--force"])
    assert result.exit_code == 0
    assert seen["force"] is True
    assert "7" in result.stdout
    assert "local use only" in result.stdout
