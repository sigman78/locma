from __future__ import annotations

from fastapi.testclient import TestClient

from locma.server.app import create_app


def _client(tmp_path):
    app = create_app(
        replay_dir=str(tmp_path / "replays"),
        asset_dir=str(tmp_path / "assets"),
        gamelog_dir=str(tmp_path / "logs"),
    )
    return TestClient(app)


def test_cards_endpoint(tmp_path):
    r = _client(tmp_path).get("/api/cards")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 160 and data[0]["name"] == "Slimer"


def test_policies_endpoint(tmp_path):
    r = _client(tmp_path).get("/api/policies")
    assert r.status_code == 200
    assert r.json() == ["random", "scripted", "greedy"]
