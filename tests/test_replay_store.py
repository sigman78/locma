from __future__ import annotations

import json

import pytest

from locma.harness.replay_store import get_replay, list_headers, write_replay

# ---------------------------------------------------------------------------
# Helpers — build a realistic replay dict without running the game engine
# ---------------------------------------------------------------------------

_SNAPSHOT = {
    "current": 0,
    "players": [
        {
            "health": 30,
            "mana": 3,
            "max_mana": 3,
            "damage_counter": 0,
            "bonus_draw": 0,
            "deck_count": 20,
            "hand": [],
            "board": [],
        },
        {
            "health": 28,
            "mana": 2,
            "max_mana": 3,
            "damage_counter": 1,
            "bonus_draw": 0,
            "deck_count": 18,
            "hand": [],
            "board": [],
        },
    ],
}


def _snap(current=0):
    s = json.loads(json.dumps(_SNAPSHOT))  # deep copy
    s["current"] = current
    return s


def _realistic_replay(rid="r_test001", created="2026-06-24T12:00:00Z"):
    """
    Draft: 2 rounds, each a 3-card pool; two picks per round (seat 0 and 1).
    Battle: opening snapshot + multiple steps where turn 2 has TWO actions
            from seat 0 sharing the same (seat=0, turn=2) — exercises the
            consecutive-run grouping.
    """
    header = {
        "replay_id": rid,
        "created_at": created,
        "source": "test",
        "format": "locma-replay/1",
        "engine_version": "0+test",
        "policy_a": "random",
        "policy_b": "random",
        "seed": 42,
        "a_seat": 0,
        "winner": 0,
        "turns": 3,
        "step_count": 4,
        "hash": "sha256:aabbccddeeff0011",
    }
    draft = {
        "pool": [[1, 2, 3], [4, 5, 6]],
        "picks": [
            {"round": 0, "seat": 0, "pick": 1},
            {"round": 0, "seat": 1, "pick": 2},
            {"round": 1, "seat": 0, "pick": 4},
            {"round": 1, "seat": 1, "pick": 5},
        ],
    }
    steps = [
        # turn 1, seat 0 — single action
        {"seat": 0, "turn": 1, "action": {"t": "pass"}, "state": _snap(0)},
        # turn 2, seat 1 — single action
        {"seat": 1, "turn": 2, "action": {"t": "pass"}, "state": _snap(1)},
        # turn 2, seat 0 — TWO consecutive actions sharing (seat=0, turn=2)
        {
            "seat": 0,
            "turn": 2,
            "action": {"t": "summon", "iid": 10},
            "state": _snap(0),
        },
        {
            "seat": 0,
            "turn": 2,
            "action": {"t": "attack", "attacker": 10, "target": -1},
            "state": _snap(0),
        },
    ]
    battle = {"opening": _snap(0), "steps": steps}
    result = {"winner": 0, "turns": 3}
    return {
        "header": header,
        "draft": draft,
        "battle": battle,
        "result": result,
    }


# ---------------------------------------------------------------------------
# Round-trip: full equality
# ---------------------------------------------------------------------------


def test_roundtrip_full_equality(tmp_path):
    original = _realistic_replay()
    write_replay(str(tmp_path), original)
    got = get_replay(str(tmp_path), "r_test001")
    assert got == original


# ---------------------------------------------------------------------------
# write_replay: path ends in .jsonl, no .meta.json written
# ---------------------------------------------------------------------------


def test_write_returns_jsonl_path(tmp_path):
    path = write_replay(str(tmp_path), _realistic_replay())
    assert path.endswith(".jsonl"), f"Expected .jsonl path, got: {path}"


def test_write_does_not_create_meta_json(tmp_path):
    write_replay(str(tmp_path), _realistic_replay())
    meta_files = list(tmp_path.glob("*.meta.json"))
    assert meta_files == [], f"Unexpected .meta.json files: {meta_files}"


# ---------------------------------------------------------------------------
# list_headers: sorted desc, corrupt-file tolerance
# ---------------------------------------------------------------------------


def test_list_headers_sorted_desc(tmp_path):
    write_replay(str(tmp_path), _realistic_replay("r_old", "2026-01-01T00:00:00Z"))
    write_replay(str(tmp_path), _realistic_replay("r_new", "2026-06-24T12:00:00Z"))
    heads = list_headers(str(tmp_path))
    assert [h["replay_id"] for h in heads] == ["r_new", "r_old"]


def test_list_headers_tolerates_corrupt_first_line(tmp_path):
    write_replay(str(tmp_path), _realistic_replay("r_ok", "2026-03-01T00:00:00Z"))
    # Write a .jsonl whose first line is not valid JSON
    (tmp_path / "r_bad.jsonl").write_text("{not json\n{also bad}\n", encoding="utf-8")
    heads = list_headers(str(tmp_path))
    assert [h["replay_id"] for h in heads] == ["r_ok"]


def test_list_headers_tolerates_empty_file(tmp_path):
    write_replay(str(tmp_path), _realistic_replay("r_ok", "2026-03-01T00:00:00Z"))
    (tmp_path / "r_empty.jsonl").write_text("", encoding="utf-8")
    heads = list_headers(str(tmp_path))
    assert [h["replay_id"] for h in heads] == ["r_ok"]


# ---------------------------------------------------------------------------
# get_replay: missing id raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_get_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_replay(str(tmp_path), "r_nope")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_draft_roundtrip(tmp_path):
    """pool=None → zero draft lines written; round-trips as pool=None, picks=[]."""
    original = _realistic_replay()
    original["draft"] = {"pool": None, "picks": []}
    write_replay(str(tmp_path), original)
    got = get_replay(str(tmp_path), original["header"]["replay_id"])
    assert got["draft"]["pool"] is None
    assert got["draft"]["picks"] == []
    assert got == original


def test_empty_battle_steps_roundtrip(tmp_path):
    """No battle steps → opening only; steps round-trip as []."""
    original = _realistic_replay("r_empty_steps", "2026-06-24T09:00:00Z")
    original["battle"]["steps"] = []
    write_replay(str(tmp_path), original)
    got = get_replay(str(tmp_path), "r_empty_steps")
    assert got["battle"]["steps"] == []
    assert got == original


def test_no_opening_roundtrip(tmp_path):
    """opening=None → open line omitted; round-trips as opening=None."""
    original = _realistic_replay("r_no_open", "2026-06-24T08:00:00Z")
    original["battle"]["opening"] = None
    write_replay(str(tmp_path), original)
    got = get_replay(str(tmp_path), "r_no_open")
    assert got["battle"]["opening"] is None
    assert got == original


def test_nonconsecutive_same_seat_turn_preserved(tmp_path):
    """Steps with the same (seat, turn) key that are NON-CONSECUTIVE must be
    stored as separate turn runs and round-trip in their original order.

    A correct consecutive-run groupby emits three turn lines for the three
    steps below; a wrong global dict accumulator would merge steps 0 and 2
    into a single turn line (two actions) and corrupt order.
    """
    original = _realistic_replay("r_nonconsec", "2026-06-24T07:00:00Z")
    # Override steps: (seat=0,turn=1), (seat=1,turn=1), (seat=0,turn=1) — same
    # key for steps 0 and 2 but they are separated by step 1.
    original["battle"]["steps"] = [
        {"seat": 0, "turn": 1, "action": {"t": "pass"}, "state": _snap(0)},
        {"seat": 1, "turn": 1, "action": {"t": "pass"}, "state": _snap(1)},
        {
            "seat": 0,
            "turn": 1,
            "action": {"t": "attack", "attacker": 5, "target": -1},
            "state": _snap(0),
        },
    ]
    write_replay(str(tmp_path), original)
    got = get_replay(str(tmp_path), "r_nonconsec")
    assert got["battle"]["steps"] == original["battle"]["steps"]
    assert got == original
