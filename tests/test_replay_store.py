from __future__ import annotations

import pytest

from locma.harness.replay_store import get_replay, list_headers, write_replay


def _replay(rid, created):
    return {"header": {"replay_id": rid, "created_at": created, "winner": 0},
            "draft": {}, "battle": {}, "result": {}}


def test_write_then_get_roundtrip(tmp_path):
    path = write_replay(str(tmp_path), _replay("r_aaa", "2026-01-01T00:00:00Z"))
    assert path.endswith("r_aaa.json")
    got = get_replay(str(tmp_path), "r_aaa")
    assert got["header"]["replay_id"] == "r_aaa"


def test_list_headers_sorted_desc(tmp_path):
    write_replay(str(tmp_path), _replay("r_old", "2026-01-01T00:00:00Z"))
    write_replay(str(tmp_path), _replay("r_new", "2026-06-01T00:00:00Z"))
    heads = list_headers(str(tmp_path))
    assert [h["replay_id"] for h in heads] == ["r_new", "r_old"]


def test_get_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_replay(str(tmp_path), "r_nope")
