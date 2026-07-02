from __future__ import annotations

import pytest

from locma.harness.card_impact import (
    estimate_card_impact,
    read_card_impact_weights,
    sweep_impact_drafts,
    write_card_impact,
)


def test_estimate_card_impact_returns_all_cards():
    report = estimate_card_impact(games=4, seed=0, battle="ground", alpha=1.0)
    assert report.games == 4
    assert report.battle == "ground"
    assert len(report.rows) == 160
    assert all(row.coefficient == row.coefficient for row in report.rows)


def test_estimate_card_impact_validates_inputs():
    with pytest.raises(ValueError):
        estimate_card_impact(games=0)
    with pytest.raises(ValueError):
        estimate_card_impact(games=1, alpha=-1)
    with pytest.raises(ValueError):
        estimate_card_impact(games=1, battle="missing")


def test_estimate_card_impact_accepts_policy_specs():
    report = estimate_card_impact(games=2, seed=1, battle="max-guard", alpha=1.0)
    assert report.battle == "max-guard"
    assert len(report.rows) == 160


def test_sweep_impact_drafts_returns_candidate_rows():
    sweep = sweep_impact_drafts(
        battle="ground",
        fit_games=4,
        fit_seed=0,
        eval_games=2,
        eval_seed=10,
        specs=[(1.0, 2.0, 3.0)],
    )
    assert sweep.fit_report.games == 4
    assert len(sweep.rows) == 1
    assert sweep.rows[0].games == 4


def test_card_impact_artifact_round_trip(tmp_path):
    report = estimate_card_impact(games=4, seed=0, battle="ground", alpha=1.0)
    path = tmp_path / "impact.json"
    write_card_impact(report, path)
    weights = read_card_impact_weights(path)
    assert len(weights) == 160
    assert set(weights) == set(range(1, 161))


def test_read_card_impact_weights_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        read_card_impact_weights(tmp_path / "missing.json")
