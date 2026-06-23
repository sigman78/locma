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


def test_run_list_get_replay(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/replays", json={"policy_a": "random", "policy_b": "random", "seed": 1})
    assert r.status_code == 200
    header = r.json()
    rid = header["replay_id"]
    assert header["policy_a"] == "random"

    idx = c.get("/api/replays").json()
    assert any(h["replay_id"] == rid for h in idx)

    full = c.get(f"/api/replays/{rid}").json()
    assert full["header"]["replay_id"] == rid
    assert full["battle"]["opening"] is not None

    assert c.get("/api/replays/r_missing").status_code == 404


def test_run_unknown_policy_returns_400(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/replays", json={"policy_a": "nonexistent", "policy_b": "random", "seed": 0})
    assert r.status_code == 400
