"""Tests for the web panel's depot API (list/show/pin/pull/publish/gc/resolve)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from locma.depot import publish
from locma.server.app import create_app


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("LOCMA_DEPOT", str(tmp_path / "depot"))
    monkeypatch.setenv("LOCMA_DEPOT_REMOTE", f"dir:{tmp_path / 'share'}")
    return TestClient(
        create_app(
            replay_dir=str(tmp_path / "replays"),
            asset_dir=str(tmp_path / "assets"),
            gamelog_dir=str(tmp_path / "logs"),
            presets_dir=str(tmp_path / "presets"),
            results_dir=str(tmp_path / "results"),
            workers=1,
        )
    )


def _seed_artifact(tmp_path, name="m", content=b"payload"):
    f = tmp_path / "a.zip"
    f.write_bytes(content)
    return publish(name, [f], note="seeded", root=tmp_path / "depot")


def test_list_and_show(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/depot").json() == []
    _seed_artifact(tmp_path)
    listed = c.get("/api/depot").json()
    assert len(listed) == 1 and listed[0]["name"] == "m" and listed[0]["pin"] == 1
    assert listed[0]["versions"][0]["status"] == "local"
    assert listed[0]["versions"][0]["size"] == len(b"payload")

    assert c.get("/api/depot/m").status_code == 200
    assert c.get("/api/depot/nope").status_code == 404


def test_pin(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    _seed_artifact(tmp_path)
    f = tmp_path / "a.zip"
    f.write_bytes(b"v2")
    publish("m", [f], root=tmp_path / "depot")

    r = c.post("/api/depot/m/pin", json={"version": 1})
    assert r.status_code == 200 and r.json()["pin"] == 1
    assert c.post("/api/depot/m/pin", json={"version": 99}).status_code == 400


def test_push_pull_roundtrip_via_dir_remote(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    vrec = _seed_artifact(tmp_path)

    r = c.post("/api/depot/m/push", json={})
    assert r.status_code == 200 and r.json()["locator"].startswith("dir:")

    # drop the local blob, then pull it back through the API
    from locma.depot import core  # noqa: PLC0415

    blob = core.blob_path(tmp_path / "depot", vrec["files"]["a.zip"]["sha256"], "a.zip")
    blob.unlink()
    r = c.post("/api/depot/m/pull", json={})
    assert r.status_code == 200 and r.json()["fetched"] == ["a.zip"]
    assert blob.is_file()
    # second pull is a no-op
    assert c.post("/api/depot/m/pull", json={}).json()["fetched"] == []


def test_publish_endpoint(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    f = tmp_path / "model.zip"
    f.write_bytes(b"weights")
    r = c.post(
        "/api/depot/publish",
        json={
            "name": "web-model",
            "files": [str(f)],
            "kind": "model",
            "note": "from the panel",
            "meta": {"avg_hard3": 0.5},
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 1
    rec = c.get("/api/depot/web-model").json()
    assert rec["versions"][0]["note"] == "from the panel"
    assert rec["versions"][0]["meta"]["avg_hard3"] == 0.5

    # bad publish: missing file
    r = c.post("/api/depot/publish", json={"name": "x", "files": [str(tmp_path / "no.zip")]})
    assert r.status_code == 400


def test_gc_and_resolve(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    _seed_artifact(tmp_path)
    f = tmp_path / "a.zip"
    f.write_bytes(b"v2")
    publish("m", [f], root=tmp_path / "depot")  # pin -> v2, v1 unreachable

    r = c.post("/api/depot/gc", json={"dry_run": True})
    assert r.status_code == 200 and r.json()["removed"] == 1 and r.json()["dry_run"]
    r = c.post("/api/depot/gc", json={"dry_run": False})
    assert r.json()["removed"] == 1

    r = c.get("/api/depot/resolve/depot:m/a.zip")
    assert r.status_code == 200 and r.json()["path"].endswith("a.zip")
    assert c.get("/api/depot/resolve/depot:m@1/a.zip").status_code == 400  # gc'd blob
    # raw paths pass through
    assert c.get("/api/depot/resolve/runs/x.zip").json()["path"] == "runs/x.zip"


def test_remote_endpoint(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/depot/remote").json()["remote"].startswith("dir:")
