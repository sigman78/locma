from __future__ import annotations

import json

from fastapi.testclient import TestClient

from locma.harness.replay_stream import build_replay
from locma.policies.registry import make_policy
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


def _write_log(tmp_path):
    rep = build_replay(make_policy("greedy"), make_policy("random"), seed=2, created_at="t")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "policy_a": "greedy",
        "policy_b": "random",
        "seed": 2,
        "a_seat": 0,
        "hash": rep["header"]["hash"],
    }
    (logs_dir / "g.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return "g.jsonl"


def test_game_logs_and_import(tmp_path):
    name = _write_log(tmp_path)
    c = _client(tmp_path)
    logs = c.get("/api/game-logs").json()
    assert logs == [{"path": name, "rows": 1}]
    r = c.post("/api/replays/import", json={"path": name, "row": 0})
    assert r.status_code == 200
    assert r.json()["source"] == f"game-log:{name}#0"


def test_art_endpoint(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    # 1x1 transparent PNG bytes
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
        "00000049454e44ae426082"
    )
    (assets / "001.png").write_bytes(png)
    c = _client(tmp_path)
    assert c.get("/api/art/1").status_code == 200
    assert c.get("/api/art/1").headers["content-type"] == "image/png"
    assert c.get("/api/art/999").status_code == 404
