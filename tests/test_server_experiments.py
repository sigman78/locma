"""Tests for the web panel's experiments API (kinds, presets, background jobs)."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from locma.server.app import create_app


def _client(tmp_path) -> TestClient:
    return TestClient(
        create_app(
            replay_dir=str(tmp_path / "replays"),
            asset_dir=str(tmp_path / "assets"),
            gamelog_dir=str(tmp_path / "logs"),
            presets_dir=str(tmp_path / "presets"),
            results_dir=str(tmp_path / "results"),
            workers=1,  # cells run inline on the collector thread: fast, no pool
        )
    )


def _wait(c: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = c.get(f"/api/experiments/jobs/{job_id}").json()
        if j["state"] not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_kinds_schema(tmp_path):
    r = _client(tmp_path).get("/api/experiments/kinds")
    assert r.status_code == 200
    kinds = {k["kind"]: k for k in r.json()}
    assert set(kinds) == {"match", "noise-floor", "league", "ceiling", "train-zoo"}
    for k in kinds.values():
        assert k["label"] and k["schema"]
        for f in k["schema"]:
            assert {"name", "type", "default"} <= set(f)


def test_match_job_end_to_end(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={
            "kind": "match",
            "params": {"policy_a": "greedy", "policy_b": "random", "games": 4, "seed": 0},
            "name": "smoke",
        },
    )
    assert r.status_code == 200, r.text
    job = _wait(c, r.json()["job_id"])
    assert job["state"] == "done", job.get("error")
    res = job["result"]
    assert res["games"] == 8 and 0.0 <= res["win_rate"] <= 1.0
    assert res["ci_lo"] <= res["win_rate"] <= res["ci_hi"]
    assert job["progress_done"] == job["progress_total"] == 1  # 4 pairs = 1 chunk

    # persisted for history
    assert (tmp_path / "results" / f"{job['job_id']}.json").is_file()
    listed = c.get("/api/experiments/jobs").json()
    assert any(j["job_id"] == job["job_id"] for j in listed)


def test_league_job(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={
            "kind": "league",
            "params": {
                "policies": ["random", "greedy", "scripted"],
                "games": 2,
                "seed": 0,
                "reference": "random",
            },
        },
    )
    assert r.status_code == 200, r.text
    job = _wait(c, r.json()["job_id"])
    assert job["state"] == "done", job.get("error")
    res = job["result"]
    assert [row["policy"] for row in res["table"]]  # sorted by openskill
    assert res["matrix"]["random"]["greedy"] + res["matrix"]["greedy"]["random"] == 1.0
    ref_row = next(row for row in res["table"] if row["policy"] == "random")
    assert ref_row["p_vs_ref"] is None  # no p-value against itself


def test_ceiling_job_without_ml(tmp_path):
    """Ceiling cells accept any registry spec, so the harness is testable
    without a model artifact or the [ml] extra."""
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={
            "kind": "ceiling",
            "params": {
                "candidates": ["greedy"],
                "baselines": ["random"],
                "seeds": 2,
                "games_per_seed": 2,
                "opponents": ["scripted"],
                "threshold": 0.03,
            },
        },
    )
    assert r.status_code == 200, r.text
    job = _wait(c, r.json()["job_id"])
    assert job["state"] == "done", job.get("error")
    res = job["result"]
    assert res["verdict"] in ("headroom", "ceiling-confirmed")
    assert res["cand_avg"] > res["b0_avg"]  # greedy beats random against scripted
    assert job["progress_total"] == 4  # 2 specs x 2 seeds


def test_noise_floor_job(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={"kind": "noise-floor", "params": {"policy": "random", "games": 4, "seed": 0}},
    )
    job = _wait(c, r.json()["job_id"])
    assert job["state"] == "done", job.get("error")
    assert job["result"]["resolution"] > 0


def test_bad_run_requests(tmp_path):
    c = _client(tmp_path)
    assert c.post("/api/experiments/run", json={"kind": "nope", "params": {}}).status_code == 400
    r = c.post(
        "/api/experiments/run",
        json={"kind": "match", "params": {"bogus": 1}},
    )
    assert r.status_code == 400 and "unknown params" in r.text


def test_bad_policy_spec_becomes_job_error(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={"kind": "match", "params": {"policy_a": "nope", "policy_b": "random", "games": 2}},
    )
    assert r.status_code == 200  # specs validate lazily, in the cell
    job = _wait(c, r.json()["job_id"])
    assert job["state"] == "error" and "nope" in job["error"]


def test_cancel_job(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={
            "kind": "match",
            "params": {"policy_a": "random", "policy_b": "random", "games": 2000, "seed": 0},
        },
    )
    job_id = r.json()["job_id"]
    assert c.post(f"/api/experiments/jobs/{job_id}/cancel").status_code == 200
    job = _wait(c, job_id)
    assert job["state"] == "cancelled"
    assert job["progress_done"] < job["progress_total"]


def test_presets_crud(tmp_path):
    c = _client(tmp_path)
    body = {
        "name": "Quick match",
        "kind": "match",
        "params": {"policy_a": "greedy", "policy_b": "random", "games": 10, "seed": 0},
        "note": "smoke",
    }
    r = c.put("/api/experiments/presets/quick-match", json=body)
    assert r.status_code == 200 and r.json()["id"] == "quick-match"

    listed = c.get("/api/experiments/presets").json()
    assert len(listed) == 1 and listed[0]["name"] == "Quick match"
    assert listed[0]["params"]["games"] == 10

    # invalid: unknown kind, bad id, bad params
    assert c.put("/api/experiments/presets/x", json=dict(body, kind="nope")).status_code == 400
    assert c.put("/api/experiments/presets/BAD ID", json=body).status_code == 400
    bad = dict(body, params={"bogus": 1})
    assert c.put("/api/experiments/presets/x", json=bad).status_code == 400

    assert c.delete("/api/experiments/presets/quick-match").status_code == 200
    assert c.get("/api/experiments/presets").json() == []
    assert c.delete("/api/experiments/presets/quick-match").status_code == 404


def test_match_series_stream(tmp_path):
    """Per-cell streaming: running win rate + CI band, one point per chunk."""
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={
            "kind": "match",
            "params": {"policy_a": "greedy", "policy_b": "random", "games": 25, "seed": 0},
        },
    )
    job = _wait(c, r.json()["job_id"])
    assert job["state"] == "done", job.get("error")
    assert "series" not in job  # jobs endpoints stay slim

    s = c.get(f"/api/experiments/jobs/{job['job_id']}/series").json()["series"]
    assert set(s) == {"win_rate", "ci_lo", "ci_hi"}
    assert len(s["win_rate"]) == 3  # 25 pairs = 3 chunks
    last_x, last_y = s["win_rate"][-1]
    assert last_x == 50  # 25 pairs mirrored
    assert abs(last_y - job["result"]["win_rate"]) < 1e-9
    for (_, lo), (_, wr), (_, hi) in zip(s["ci_lo"], s["win_rate"], s["ci_hi"], strict=True):
        assert lo <= wr <= hi


def test_ceiling_series_stream(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={
            "kind": "ceiling",
            "params": {
                "candidates": ["greedy"],
                "baselines": ["random"],
                "seeds": 3,
                "games_per_seed": 2,
                "opponents": ["scripted"],
            },
        },
    )
    job = _wait(c, r.json()["job_id"])
    assert job["state"] == "done", job.get("error")
    s = c.get(f"/api/experiments/jobs/{job['job_id']}/series").json()["series"]
    assert len(s["delta"]) == 3  # one paired delta per seed
    assert len(s["mean_delta"]) == 3
    # the final running mean equals the reduce's mean delta
    assert abs(s["mean_delta"][-1][1] - job["result"]["mean_delta"]) < 1e-9


def test_league_live_matrix(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={
            "kind": "league",
            "params": {"policies": ["random", "greedy", "scripted"], "games": 2, "seed": 0},
        },
    )
    job = _wait(c, r.json()["job_id"])
    assert job["state"] == "done", job.get("error")
    live = c.get(f"/api/experiments/jobs/{job['job_id']}/series").json()["live"]
    assert live["policies"] == ["random", "greedy", "scripted"]
    assert live["matrix"] == job["result"]["matrix"]  # fully filled at the end


def test_series_survive_restart(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={"kind": "match", "params": {"policy_a": "random", "policy_b": "random", "games": 4}},
    )
    job = _wait(c, r.json()["job_id"])

    c2 = _client(tmp_path)  # fresh app, same results_dir: history only
    listed = c2.get("/api/experiments/jobs").json()
    doc = next(j for j in listed if j["job_id"] == job["job_id"])
    assert "series" not in doc  # stripped from the list
    s = c2.get(f"/api/experiments/jobs/{job['job_id']}/series").json()["series"]
    assert s["win_rate"], "series must be readable from the persisted job"


def test_log_endpoint_and_error_traceback(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={"kind": "match", "params": {"policy_a": "nope", "policy_b": "random", "games": 2}},
    )
    job = _wait(c, r.json()["job_id"])
    assert job["state"] == "error"
    log = c.get(f"/api/experiments/jobs/{job['job_id']}/log").json()["log"]
    assert "Traceback" in log and "nope" in log
    assert c.get("/api/experiments/jobs/nope/log").status_code == 404


def test_runner_tail_follows_metrics_file(tmp_path):
    """The tail channel: a cell streams JSONL; numeric fields become series and
    the x field drives progress (the training-job mechanism, sans SB3)."""
    from locma.server.jobs import JobRunner, TailConfig  # noqa: PLC0415

    metrics = tmp_path / "metrics.jsonl"
    runner = JobRunner(results_dir=str(tmp_path / "results"), workers=1)
    job = runner.submit(
        kind="t",
        name="t",
        params={},
        cells=[(_write_metrics_cell, (str(metrics),))],
        reduce_fn=lambda p, r: {"ok": True},
        tail=TailConfig(path=str(metrics), x="timesteps", total=300),
    )
    deadline = time.time() + 30
    while job.state in ("queued", "running") and time.time() < deadline:
        time.sleep(0.05)
    assert job.state == "done"
    assert job.progress_total == 300
    assert job.progress_done == 300
    assert job.series["ep_rew_mean"] == [[100.0, 0.1], [200.0, 0.5], [300.0, 0.9]]
    assert "note" not in job.series  # non-numeric fields are ignored


def _write_metrics_cell(path: str) -> dict:
    import json as _json  # noqa: PLC0415
    import time as _time  # noqa: PLC0415

    for ts, rew in ((100, 0.1), (200, 0.5), (300, 0.9)):
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps({"timesteps": ts, "ep_rew_mean": rew, "note": "x"}) + "\n")
        _time.sleep(0.05)
    return {}


@pytest.mark.slow
def test_train_zoo_job_slow(tmp_path):
    """End-to-end training job: metrics stream -> series + progress; model saved."""
    pytest.importorskip("sb3_contrib")
    c = _client(tmp_path)
    r = c.post(
        "/api/experiments/run",
        json={
            "kind": "train-zoo",
            "params": {
                "opponents": ["random"],
                "steps_per_opponent": 300,
                "obs_mode": "flat",
                "n_steps": 64,
                "seed": 0,
            },
        },
    )
    assert r.status_code == 200, r.text
    job = _wait(c, r.json()["job_id"], timeout=300)
    assert job["state"] == "done", job.get("error")
    assert job["result"]["cancelled"] is False
    assert os.path.isfile(job["result"]["out"])
    assert job["progress_total"] == 300
    assert job["progress_done"] > 0
    s = c.get(f"/api/experiments/jobs/{job['job_id']}/series").json()["series"]
    assert s.get("ep_rew_mean"), "training must stream reward points"


def test_policy_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCMA_DEPOT", str(tmp_path / "empty-depot"))
    c = _client(tmp_path)
    r = c.get("/api/policy-catalog")
    assert r.status_code == 200
    cat = r.json()
    assert "greedy" in cat["baselines"]
    assert any(b["base"] == "vbeam" for b in cat["model_bases"])
    assert cat["depot_models"] == []  # empty depot root
    assert "greedy" in cat["suggestions"]
