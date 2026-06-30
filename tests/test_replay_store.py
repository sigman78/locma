from __future__ import annotations

import json

import pytest

from locma.data.cards_db import load_cards as _load_cards
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
        "format": "locma-replay/2",
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
        {"seat": 0, "turn": 1, "action": {"t": "pass"}, "state": _snap(0), "events": []},
        # turn 2, seat 1 — single action
        {"seat": 1, "turn": 2, "action": {"t": "pass"}, "state": _snap(1), "events": []},
        # turn 2, seat 0 — TWO consecutive actions sharing (seat=0, turn=2)
        {
            "seat": 0,
            "turn": 2,
            "action": {"t": "summon", "iid": 10},
            "state": _snap(0),
            "events": [],
        },
        {
            "seat": 0,
            "turn": 2,
            "action": {"t": "attack", "attacker": 10, "target": -1},
            "state": _snap(0),
            "events": [],
        },
    ]
    battle = {"opening": _snap(0), "steps": steps, "closing": _snap(1)}
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


def test_no_closing_roundtrip(tmp_path):
    """closing=None → close line omitted; round-trips as closing=None."""
    original = _realistic_replay("r_no_close", "2026-06-24T06:00:00Z")
    original["battle"]["closing"] = None
    write_replay(str(tmp_path), original)
    got = get_replay(str(tmp_path), "r_no_close")
    assert got["battle"]["closing"] is None
    assert got == original


def _make_replay_with_steps(steps, rid="r_evtest", created="2026-06-24T12:00:00Z"):
    """Thin wrapper around _realistic_replay that overrides battle steps."""
    rep = _realistic_replay(rid, created)
    rep["battle"]["steps"] = steps
    rep["header"]["step_count"] = len(steps)
    return rep


def test_events_roundtrip(tmp_path):
    rep = _make_replay_with_steps(
        [
            {
                "seat": 0,
                "turn": 1,
                "action": {"t": "pass"},
                "state": _snap(0),
                "events": [
                    {"t": "turn_ended", "seat": 0},
                    {"t": "turn_started", "seat": 1, "draws": [10]},
                ],
            }
        ]
    )
    write_replay(str(tmp_path), rep)
    got = get_replay(str(tmp_path), rep["header"]["replay_id"])
    assert got["battle"]["steps"][0]["events"] == rep["battle"]["steps"][0]["events"]


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
        {"seat": 0, "turn": 1, "action": {"t": "pass"}, "state": _snap(0), "events": []},
        {"seat": 1, "turn": 1, "action": {"t": "pass"}, "state": _snap(1), "events": []},
        {
            "seat": 0,
            "turn": 1,
            "action": {"t": "attack", "attacker": 5, "target": -1},
            "state": _snap(0),
            "events": [],
        },
    ]
    write_replay(str(tmp_path), original)
    got = get_replay(str(tmp_path), "r_nonconsec")
    assert got["battle"]["steps"] == original["battle"]["steps"]
    assert got == original


_CAT = {c.id: c for c in _load_cards()}


def _v3_replay(rid="r_v3"):
    rep = _realistic_replay(rid)
    rep["header"]["format"] = "locma-replay/3"
    return rep


def _read_lines(tmp_path, rid):
    text = (tmp_path / f"{rid}.jsonl").read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def test_v3_emits_phase_framing_and_deltas(tmp_path):
    write_replay(str(tmp_path), _v3_replay())
    kinds = [ln["k"] for ln in _read_lines(tmp_path, "r_v3")]
    assert kinds.count("draft_start") == 1 and kinds.count("draft_end") == 1
    assert "battle_start" in kinds and "battle_end" in kinds
    assert "open" not in kinds and "close" not in kinds
    # turn actions carry "d" deltas, never a full "state"
    for ln in _read_lines(tmp_path, "r_v3"):
        if ln["k"] == "turn":
            for a in ln["actions"]:
                assert "d" in a and "state" not in a


def test_v2_path_unchanged(tmp_path):
    # an explicit /2 dict must still emit the legacy open/turn-with-state lines
    write_replay(str(tmp_path), _realistic_replay("r_v2"))
    kinds = [ln["k"] for ln in _read_lines(tmp_path, "r_v2")]
    assert "open" in kinds and "draft_start" not in kinds
    for ln in _read_lines(tmp_path, "r_v2"):
        if ln["k"] == "turn":
            assert all("state" in a for a in ln["actions"])


def _v3_with_cards(rid="r_v3cards"):
    """A /3 replay whose states carry base, buffed, damaged and keyword-changed cards."""

    def hc(iid, cid):
        c = _CAT[cid]
        return {
            "iid": iid,
            "card_id": cid,
            "atk": c.attack,
            "def": c.defense,
            "abilities": c.abilities,
        }

    def bc(iid, cid, **over):
        d = hc(iid, cid)
        d["can_attack"] = True
        d["has_attacked"] = False
        d.update(over)
        return d

    def state(current, h0, b0, h1, b1):
        def pl(hand, board):
            return {
                "health": 30,
                "mana": 2,
                "max_mana": 2,
                "damage_counter": 0,
                "bonus_draw": 0,
                "deck_count": 20,
                "hand": hand,
                "board": board,
            }

        return {"current": current, "players": [pl(h0, b0), pl(h1, b1)]}

    buffed = bc(20, 3)
    buffed["atk"] = _CAT[3].attack + 2
    buffed["def"] = _CAT[3].defense + 2
    damaged = bc(21, 5)
    damaged["def"] = max(0, _CAT[5].defense - 1)
    guarded = bc(22, 9)
    guarded["abilities"] = "---G--"

    opening = state(0, [hc(10, 3), hc(11, 5)], [], [hc(12, 9)], [])
    s1 = state(0, [hc(11, 5)], [buffed], [hc(12, 9)], [])  # seat0 summoned+buffed 3
    s2 = state(1, [hc(11, 5)], [buffed], [], [guarded, damaged])  # seat1 board changed
    closing = state(1, [hc(11, 5)], [buffed], [], [guarded])

    rep = _realistic_replay(rid)
    rep["header"]["format"] = "locma-replay/3"
    rep["battle"] = {
        "opening": opening,
        "steps": [
            {"seat": 0, "turn": 1, "action": {"t": "summon", "iid": 20}, "state": s1, "events": []},
            {
                "seat": 1,
                "turn": 2,
                "action": {"t": "pass"},
                "state": s2,
                "events": [{"t": "turn_started", "seat": 1, "draws": [12]}],
            },
        ],
        "closing": closing,
    }
    return rep


def test_v3_roundtrip_lossless_with_mutable_stats(tmp_path):
    original = _v3_with_cards()
    write_replay(str(tmp_path), original)
    got = get_replay(str(tmp_path), original["header"]["replay_id"])
    assert got == original


def test_v3_no_closing_roundtrip(tmp_path):
    original = _v3_with_cards("r_v3noclose")
    original["battle"]["closing"] = None
    write_replay(str(tmp_path), original)
    got = get_replay(str(tmp_path), "r_v3noclose")
    assert got["battle"]["closing"] is None
    assert got == original
