"""Pure plumbing tests for the crash-resumable E24 rbeam benchmark driver."""

import pytest

from scripts.e24_rbeam_bench import (
    Cell,
    build_blocks,
    candidate_spec,
    parse_allocations,
    parse_confirm_cell,
    planner_spec,
    select_confirm_cells,
)


def test_e24_allocations_parse_plans_by_worlds():
    assert parse_allocations("2x2, 3x3,4x4") == [(2, 2), (3, 3), (4, 4)]
    # Unlike E23 there is no fixed-budget constraint: mixed products are fine.
    assert parse_allocations("2x4,4x2") == [(2, 4), (4, 2)]
    with pytest.raises(ValueError, match="n_plans >= 1"):
        parse_allocations("0x3")
    with pytest.raises(ValueError, match="duplicate"):
        parse_allocations("3x3,3x3")


def test_e24_candidate_spec_is_rbeam_ensemble_same_draft():
    spec = candidate_spec(Cell(4, 2))
    assert spec.startswith("rbeam:depot:shared/shared_s0.zip|")
    assert ",8,20,4,2," in spec  # width, max_actions, n_plans, n_worlds
    assert spec.endswith("depot:ldraft/ldraft_s0.zip")
    # Both sides share the ensemble + ldraft, isolating the reply ply.
    assert planner_spec().endswith("depot:ldraft/ldraft_s0.zip")


def test_e24_blocks_cover_seed_pairs_without_overlap():
    assert build_blocks(29_000_000, pairs=12, block_pairs=5) == [
        (0, 29_000_000, 5),
        (1, 29_000_005, 5),
        (2, 29_000_010, 2),
    ]


def test_e24_confirm_selector_prefers_cost_gate_then_win_rate():
    assert parse_confirm_cell("3x3") == Cell(3, 3)
    rows = {
        # Strongest win rate but OUTSIDE the cost gate.
        "p4_w4": {
            "n_plans": 4,
            "n_worlds": 4,
            "reply_beams": 16,
            "candidate_wr": 0.62,
            "ci_lo": 0.50,
            "games": 50,
            "within_cost_gate": False,
        },
        # Slightly weaker but WITHIN the gate -> should be picked.
        "p3_w3": {
            "n_plans": 3,
            "n_worlds": 3,
            "reply_beams": 9,
            "candidate_wr": 0.58,
            "ci_lo": 0.47,
            "games": 50,
            "within_cost_gate": True,
        },
    }
    assert select_confirm_cells(rows) == [Cell(3, 3)]


def test_e24_confirm_selector_ties_break_to_cheaper_cell():
    rows = {
        "p2_w4": {
            "n_plans": 2,
            "n_worlds": 4,
            "reply_beams": 8,
            "candidate_wr": 0.60,
            "ci_lo": 0.49,
            "games": 50,
            "within_cost_gate": True,
        },
        "p4_w2": {
            "n_plans": 4,
            "n_worlds": 2,
            "reply_beams": 8,
            "candidate_wr": 0.60,
            "ci_lo": 0.49,
            "games": 50,
            "within_cost_gate": True,
        },
        # Same win rate/CI, more beams -> loses the reply_beams tie-break.
        "p3_w3": {
            "n_plans": 3,
            "n_worlds": 3,
            "reply_beams": 9,
            "candidate_wr": 0.60,
            "ci_lo": 0.49,
            "games": 50,
            "within_cost_gate": True,
        },
    }
    winner = select_confirm_cells(rows)[0]
    assert winner.reply_beams == 8
