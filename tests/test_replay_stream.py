from __future__ import annotations

import pytest

from locma.policies.registry import make_policy


def test_make_policy_known():
    for spec in ("random", "scripted", "greedy"):
        assert make_policy(spec).name == spec


def test_make_policy_unknown():
    with pytest.raises(ValueError, match=r"unknown policy 'nope'"):
        make_policy("nope")


from locma.core.engine import run_game
from locma.harness.replay_stream import StreamRecorder
from locma.policies.random_policy import RandomPolicy


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
    state = rec.steps[0]["state"]
    p0 = state["players"][0]
    # both hands fully visible
    assert "hand" in p0 and "board" in p0
    for c in p0["hand"]:
        assert set(c) == {"iid", "card_id", "atk", "def", "abilities"}
    for c in p0["board"]:
        assert {"can_attack", "has_attacked"} <= set(c)
