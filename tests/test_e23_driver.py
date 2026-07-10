"""Pure plumbing tests for the crash-resumable E23 experiment driver."""

import pytest

from scripts.e23_netdmcts_alloc import (
    Cell,
    build_blocks,
    candidate_spec,
    parse_allocations,
    parse_confirm_cell,
    select_confirm_cells,
)


def test_e23_allocations_require_one_fixed_budget():
    assert parse_allocations("1x320, 2x160,4x80") == [(1, 320), (2, 160), (4, 80)]
    with pytest.raises(ValueError, match=r"fixed K\*I"):
        parse_allocations("1x320,2x80")
    with pytest.raises(ValueError, match="K >= 1"):
        parse_allocations("0x320")


def test_e23_candidate_spec_carries_same_draft_override():
    spec = candidate_spec(Cell("b0k", 2, 160))
    assert spec.startswith("netdmcts:2,160,1.5,depot:b0k/b0k_s0.zip,")
    assert spec.endswith("depot:ldraft/ldraft_s0.zip")


def test_e23_blocks_cover_seed_pairs_without_overlap():
    assert build_blocks(27_000_000, pairs=12, block_pairs=5) == [
        (0, 27_000_000, 5),
        (1, 27_000_005, 5),
        (2, 27_000_010, 2),
    ]


def test_e23_confirm_selector_and_pilot_ranking():
    assert parse_confirm_cell("shared:4x80") == Cell("shared", 4, 80)
    rows = {
        "a": {"oracle": "b0k", "K": 8, "I": 40, "candidate_wr": 0.52, "ci_lo": 0.42, "games": 50},
        "b": {
            "oracle": "shared",
            "K": 2,
            "I": 160,
            "candidate_wr": 0.60,
            "ci_lo": 0.50,
            "games": 50,
        },
    }
    assert select_confirm_cells(rows) == [Cell("shared", 2, 160)]
