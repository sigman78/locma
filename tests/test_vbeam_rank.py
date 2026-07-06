"""Unit tests for E15's sibling-pair construction (pure numpy, no [ml])."""

from __future__ import annotations

import numpy as np

from locma.envs.vbeam_rank import build_pairs


def _arrs(rows):
    """rows: (call, depth, stop_ok, target) tuples -> column arrays."""
    call = np.asarray([r[0] for r in rows], dtype=np.int64)
    depth = np.asarray([r[1] for r in rows], dtype=np.int64)
    stop = np.asarray([r[2] for r in rows], dtype=bool)
    tgt = np.asarray([r[3] for r in rows], dtype=np.float32)
    return call, depth, stop, tgt


def test_same_call_same_depth_pairs():
    """Depth-1 siblings of one call pair up, ordered hi-first."""
    rows = [(0, 0, False, 0.5), (0, 1, False, 0.3), (0, 1, False, 0.7)]
    hi, lo = build_pairs(*_arrs(rows))
    assert len(hi) == 1
    assert (hi[0], lo[0]) == (2, 1)  # 0.7 > 0.3


def test_root_never_pairs_with_children():
    """Depth 0 joins no depth pool (the root is not a sibling of its kids)."""
    rows = [(0, 0, False, 0.9), (0, 1, False, 0.3)]
    hi, _lo = build_pairs(*_arrs(rows))
    assert len(hi) == 0


def test_no_cross_call_pairs():
    rows = [(0, 1, False, 0.9), (1, 1, False, 0.1)]
    hi, _lo = build_pairs(*_arrs(rows))
    assert len(hi) == 0


def test_stop_ok_pairs_across_depths():
    """Stop-eligible states of one call compare across depths (the completed-
    plan ranking), even when their depth pools differ."""
    rows = [(0, 1, True, 0.2), (0, 3, True, 0.6)]
    hi, lo = build_pairs(*_arrs(rows))
    assert len(hi) == 1
    assert (hi[0], lo[0]) == (1, 0)


def test_min_margin_excludes_noise_pairs():
    rows = [(0, 1, False, 0.500), (0, 1, False, 0.505)]
    hi, _lo = build_pairs(*_arrs(rows), min_margin=0.01)
    assert len(hi) == 0


def test_max_pairs_per_group_caps():
    rows = [(0, 1, False, i * 0.1) for i in range(8)]  # 28 candidate pairs
    hi, lo = build_pairs(*_arrs(rows), max_pairs_per_group=5)
    assert len(hi) == 5
    assert all(rows[h][3] > rows[le][3] for h, le in zip(hi, lo, strict=True))


def test_dual_membership_depth_and_stop_pools():
    """A stop-eligible depth-1 state pairs in BOTH its depth pool and the
    stop pool -- the two comparisons the beam actually makes."""
    rows = [(0, 1, True, 0.2), (0, 1, False, 0.8), (0, 2, True, 0.5)]
    hi, lo = build_pairs(*_arrs(rows))
    got = {(int(h), int(le)) for h, le in zip(hi, lo, strict=True)}
    assert got == {(1, 0), (2, 0)}  # depth-1 pool pair + stop-pool pair
