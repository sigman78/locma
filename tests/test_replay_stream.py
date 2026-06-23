from __future__ import annotations

import pytest

from locma.core.engine import run_game
from locma.harness.replay_stream import StreamRecorder, build_replay, build_replay_from_log_row
from locma.policies.random_policy import RandomPolicy
from locma.policies.registry import make_policy


def test_make_policy_known():
    for spec in ("random", "scripted", "greedy"):
        assert make_policy(spec).name == spec


def test_make_policy_unknown():
    with pytest.raises(ValueError, match=r"unknown policy 'nope'"):
        make_policy("nope")


def _record(seed=1):
    rec = StreamRecorder()
    result = run_game(
        RandomPolicy("a"), RandomPolicy("b"), seed=seed,
        on_step=rec.on_step, on_snapshot=rec.on_snapshot,
    )
    return rec, result


def test_recorder_captures_draft():
    rec, _ = _record()
    assert len(rec.draft_pool) == 30
    assert all(len(trip) == 3 for trip in rec.draft_pool)
    assert len(rec.draft_picks) == 60
    assert rec.draft_picks[0] == {"round": 0, "seat": 0, "pick": rec.draft_picks[0]["pick"]}
    assert rec.draft_picks[1]["round"] == 0 and rec.draft_picks[1]["seat"] == 1
    assert rec.draft_picks[2]["round"] == 1


def test_recorder_captures_opening_and_steps():
    rec, _ = _record()
    assert rec.opening is not None
    assert len(rec.opening["players"]) == 2
    assert rec.steps, "expected at least one battle step"
    # both opening hands are full info with the exact card key set
    for p in rec.opening["players"]:
        assert p["hand"], "opening hand should be non-empty"
        for c in p["hand"]:
            assert set(c) == {"iid", "card_id", "atk", "def", "abilities"}
    # board creatures collected across all steps carry can_attack/has_attacked
    creatures = [c for s in rec.steps for pl in s["state"]["players"] for c in pl["board"]]
    assert creatures, "expected at least one creature on a board during the game"
    for c in creatures:
        assert {"can_attack", "has_attacked"} <= set(c)


def test_build_replay_structure_and_hash():
    rep = build_replay(
        RandomPolicy("a"), RandomPolicy("b"), seed=5, created_at="2026-06-23T00:00:00Z"
    )
    h = rep["header"]
    assert h["format"] == "locma-replay/1"
    assert h["policy_a"] == "a" and h["policy_b"] == "b" and h["seed"] == 5
    assert h["a_seat"] == 0
    assert h["replay_id"] == "r_" + h["hash"].split(":")[1][:12]
    assert h["step_count"] == len(rep["battle"]["steps"])
    assert rep["battle"]["opening"] is not None
    assert len(rep["draft"]["pool"]) == 30 and len(rep["draft"]["picks"]) == 60
    assert rep["result"]["winner"] in (0, 1)


def test_build_replay_winner_matches_run_game():
    rep = build_replay(RandomPolicy("a"), RandomPolicy("b"), seed=9, created_at="t")
    gr = run_game(RandomPolicy("a"), RandomPolicy("b"), seed=9)
    assert rep["result"]["winner"] == gr.winner and rep["result"]["turns"] == gr.turns


def test_build_replay_from_log_row_roundtrip():
    rep = build_replay(make_policy("greedy"), make_policy("random"), seed=4, created_at="t")
    row = {
        "policy_a": "greedy", "policy_b": "random", "seed": 4,
        "a_seat": 0, "hash": rep["header"]["hash"],
    }
    out = build_replay_from_log_row(row, source="game-log:x.jsonl#0", make_policy=make_policy)
    assert out["header"]["hash"] == row["hash"]
    assert out["header"]["source"] == "game-log:x.jsonl#0"


def test_build_replay_from_log_row_hash_mismatch():
    row = {"policy_a": "greedy", "policy_b": "random", "seed": 4, "a_seat": 0, "hash": "sha256:bad"}
    with pytest.raises(ValueError):
        build_replay_from_log_row(row, source="s", make_policy=make_policy)
