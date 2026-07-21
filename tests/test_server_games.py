import tempfile

from fastapi.testclient import TestClient

from locma.server.app import create_app


def client():
    d = tempfile.mkdtemp()
    return TestClient(create_app(replay_dir=d, asset_dir=d, gamelog_dir=d)), d


def test_create_game_returns_pending():
    c, _ = client()
    r = c.post("/api/games", json={"opponent": "random", "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["game_id"].startswith("g_")
    assert body["you"] in (0, 1)
    assert body["status"] == "awaiting_human"
    assert body["pending"]["phase"] == "draft"


def test_unknown_game_404():
    c, _ = client()
    assert c.get("/api/games/g_missing").status_code == 404


def test_bad_opponent_400():
    c, _ = client()
    assert c.post("/api/games", json={"opponent": "nope"}).status_code == 400


def test_full_game_over_http_writes_replay():
    c, d = client()
    gid = c.post("/api/games", json={"opponent": "random", "seed": 2}).json()["game_id"]
    status = "awaiting_human"
    guard = 0
    while status == "awaiting_human":
        body = c.get(f"/api/games/{gid}").json()
        pending = body["pending"]
        if pending["phase"] == "draft":
            resp = c.post(f"/api/games/{gid}/draft", json={"pick": 0}).json()
        else:
            resp = c.post(f"/api/games/{gid}/action", json={"action": {"t": "pass"}}).json()
        status = resp["status"]
        guard += 1
        assert guard < 5000
    assert resp["result"]["replay_id"].startswith("r_")
    # finished game now appears in the standard replay index
    listed = c.get("/api/replays").json()
    assert any(h["replay_id"] == resp["result"]["replay_id"] for h in listed)


def test_draft_policies_listed():
    c, _ = client()
    body = c.get("/api/draft-policies").json()
    names = [p["name"] for p in body]
    assert "balanced" in names and "greedy" in names
    assert all("label" in p for p in body)


def test_complete_draft_stages_full_deck():
    c, _ = client()
    gid = c.post("/api/games", json={"opponent": "random", "seed": 3}).json()["game_id"]
    # two manual picks, then auto-complete with the balanced draft
    c.post(f"/api/games/{gid}/draft", json={"pick": 1})
    c.post(f"/api/games/{gid}/draft", json={"pick": 2})
    resp = c.post(f"/api/games/{gid}/draft/complete", json={"policy": "balanced"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pending"]["phase"] == "battle"
    assert len(body["drafted"]) == 30


def test_complete_draft_unknown_policy_400():
    c, _ = client()
    gid = c.post("/api/games", json={"opponent": "random", "seed": 3}).json()["game_id"]
    assert c.post(f"/api/games/{gid}/draft/complete", json={"policy": "nope"}).status_code == 400


def test_complete_draft_wrong_phase_409():
    c, _ = client()
    gid = c.post("/api/games", json={"opponent": "random", "seed": 3}).json()["game_id"]
    while c.get(f"/api/games/{gid}").json()["pending"]["phase"] == "draft":
        c.post(f"/api/games/{gid}/draft", json={"pick": 0})
    assert (
        c.post(f"/api/games/{gid}/draft/complete", json={"policy": "balanced"}).status_code == 409
    )


def test_wrong_phase_409():
    c, _ = client()
    gid = c.post("/api/games", json={"opponent": "random", "seed": 2}).json()["game_id"]
    # action during draft → 409
    assert c.post(f"/api/games/{gid}/action", json={"action": {"t": "pass"}}).status_code == 409


def test_illegal_action_400():
    c, _ = client()
    gid = c.post("/api/games", json={"opponent": "random", "seed": 2}).json()["game_id"]
    # finish the draft, then attack a bogus unit
    while c.get(f"/api/games/{gid}").json()["pending"]["phase"] == "draft":
        c.post(f"/api/games/{gid}/draft", json={"pick": 0})
    r = c.post(
        f"/api/games/{gid}/action", json={"action": {"t": "attack", "a": 999999, "target": -1}}
    )
    assert r.status_code == 400
