from __future__ import annotations

import pytest

from locma.core.engine import run_game
from locma.harness.replay_stream import StreamRecorder, build_replay, build_replay_from_log_row
from locma.policies.battles import RandomBattlePolicy
from locma.policies.composer import Composer
from locma.policies.drafts import RandomDraftPolicy
from locma.policies.registry import make_policy


def _random(name):
    return Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name=name)


def test_make_policy_known():
    for spec in ("random", "scripted", "greedy"):
        assert make_policy(spec).name == spec


def test_make_policy_unknown():
    with pytest.raises(ValueError, match=r"unknown policy 'nope'"):
        make_policy("nope")


def _record(seed=1):
    rec = StreamRecorder()
    result = run_game(
        _random("a"),
        _random("b"),
        seed=seed,
        on_step=rec.on_step,
        on_snapshot=rec.on_snapshot,
        on_pre_step=rec.on_pre_step,
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


def test_every_step_state_is_in_acting_seat_perspective():
    """Decision-point invariant: each recorded step's snapshot belongs to the
    seat that took the action. A turn-ending pass must NOT carry the opponent's
    already-flipped, already-drawn start-of-turn state."""
    for seed in (1, 5, 7, 9):
        rec, _ = _record(seed)
        for i, s in enumerate(rec.steps):
            assert s["state"]["current"] == s["seat"], (
                f"seed={seed} step {i}: action={s['action']} recorded for seat "
                f"{s['seat']} but state.current={s['state']['current']}"
            )


def test_turn_numbers_are_monotonic_one_per_ply():
    """Decision-point recording keeps each step's turn equal to gs.turn BEFORE the
    action, so a whole player-turn (closing pass included) shares one turn number
    and consecutive turns never go backwards."""
    rec, _ = _record(seed=7)
    turns = [s["turn"] for s in rec.steps]
    assert turns == sorted(turns), f"turn numbers not monotonic: {turns}"
    # a pass shares the turn of the action(s) preceding it in the same run
    for a, b in zip(rec.steps, rec.steps[1:], strict=False):
        if a["seat"] == b["seat"] and b["action"].get("t") == "pass":
            assert a["turn"] == b["turn"], "closing pass must share the actor's turn"
            break
    else:
        raise AssertionError("expected an action immediately followed by its pass")


def test_build_replay_structure_and_hash():
    rep = build_replay(
        _random("a"), _random("b"), seed=5, created_at="2026-06-23T00:00:00Z"
    )
    h = rep["header"]
    assert h["format"] == "locma-replay/2"
    assert h["policy_a"] == "a" and h["policy_b"] == "b" and h["seed"] == 5
    assert h["a_seat"] == 0
    assert h["replay_id"] == "r_" + h["hash"].split(":")[1][:12]
    assert h["step_count"] == len(rep["battle"]["steps"])
    assert rep["battle"]["opening"] is not None
    assert len(rep["draft"]["pool"]) == 30 and len(rep["draft"]["picks"]) == 60
    assert rep["result"]["winner"] in (0, 1)


def test_build_replay_captures_closing_final_board():
    """The closing snapshot is the final board after the game-ending action, so a
    viewer can show the last move's result (no later step carries it)."""
    rep = build_replay(_random("a"), _random("b"), seed=5, created_at="t")
    closing = rep["battle"]["closing"]
    assert closing is not None, "a game that ends normally should record a closing board"
    loser = 1 - rep["result"]["winner"]
    assert closing["players"][loser]["health"] <= 0, "loser should be dead on the final board"


def test_build_replay_winner_matches_run_game():
    rep = build_replay(_random("a"), _random("b"), seed=9, created_at="t")
    gr = run_game(_random("a"), _random("b"), seed=9)
    assert rep["result"]["winner"] == gr.winner and rep["result"]["turns"] == gr.turns


def test_build_replay_from_log_row_roundtrip():
    rep = build_replay(make_policy("greedy"), make_policy("random"), seed=4, created_at="t")
    row = {
        "policy_a": "greedy",
        "policy_b": "random",
        "seed": 4,
        "a_seat": 0,
        "hash": rep["header"]["hash"],
    }
    out = build_replay_from_log_row(row, source="game-log:x.jsonl#0", make_policy=make_policy)
    assert out["header"]["hash"] == row["hash"]
    assert out["header"]["source"] == "game-log:x.jsonl#0"


def test_build_replay_from_log_row_hash_mismatch():
    row = {"policy_a": "greedy", "policy_b": "random", "seed": 4, "a_seat": 0, "hash": "sha256:bad"}
    with pytest.raises(ValueError):
        build_replay_from_log_row(row, source="s", make_policy=make_policy)


def test_steps_carry_events_and_pass_decomposed():
    rep = build_replay(make_policy("random"), make_policy("random"), seed=1)
    assert rep["header"]["format"] == "locma-replay/2"
    steps = rep["battle"]["steps"]
    assert all("events" in s for s in steps)
    # at least one turn-ending pass step carries turn_ended + turn_started
    pass_steps = [s for s in steps if s["action"] == {"t": "pass"}]
    assert any({"turn_ended", "turn_started"} <= {e["t"] for e in s["events"]} for s in pass_steps)
