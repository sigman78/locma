from locma.core.actions import Action
from locma.harness.trace import (
    canonical_json,
    read_game_log,
    record_game,
    serialize_trace,
    trace_hash,
    write_game_log,
)
from locma.policies.greedy import GreedyPolicy


def test_record_game_returns_trace():
    result, trace = record_game(GreedyPolicy(), GreedyPolicy(), seed=3)
    assert result.winner in (0, 1)
    assert len(trace) > 0
    seat, action = trace[0]
    assert seat in (0, 1)
    assert isinstance(action, (int, *Action.__args__))


def test_hash_is_deterministic():
    r1, t1 = record_game(GreedyPolicy(), GreedyPolicy(), seed=7)
    r2, t2 = record_game(GreedyPolicy(), GreedyPolicy(), seed=7)
    assert trace_hash(t1, r1.winner, r1.turns) == trace_hash(t2, r2.winner, r2.turns)


def test_hash_changes_with_outcome():
    r, t = record_game(GreedyPolicy(), GreedyPolicy(), seed=7)
    h = trace_hash(t, r.winner, r.turns)
    assert h != trace_hash(t, 1 - r.winner, r.turns)


def test_canonical_json_is_sorted_compact():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_serialize_trace_encodes_draft_and_battle():
    _, trace = record_game(GreedyPolicy(), GreedyPolicy(), seed=1)
    ser = serialize_trace(trace)
    tags = {entry[1]["t"] for entry in ser}
    assert "draft" in tags
    assert tags & {"summon", "attack", "use", "pass"}


def test_game_log_roundtrip(tmp_path):
    path = tmp_path / "g.jsonl"
    rec = {"format": 1, "seed": 1, "winner": 0, "turns": 5, "hash": "sha256:abc"}
    write_game_log(str(path), [rec])
    assert read_game_log(str(path)) == [rec]
