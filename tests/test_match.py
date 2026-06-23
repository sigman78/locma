from __future__ import annotations

import json

from locma.harness.match import run_match
from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy


def test_run_match_counts_and_balances_sides():
    res = run_match(RandomPolicy("a"), RandomPolicy("b"), games=10, seed=0)
    assert res.games == 20  # mirrored pairs => 2 games each
    assert res.wins_a + res.wins_b == 20
    assert 0.0 <= res.win_rate_a <= 1.0


def test_match_is_deterministic():
    r1 = run_match(RandomPolicy("a"), RandomPolicy("b"), games=8, seed=5)
    r2 = run_match(RandomPolicy("a"), RandomPolicy("b"), games=8, seed=5)
    assert (r1.wins_a, r1.wins_b) == (r2.wins_a, r2.wins_b)


def test_jsonl_written(tmp_path):
    p = tmp_path / "out.jsonl"
    run_match(ScriptedPolicy("s"), RandomPolicy("r"), games=3, seed=1, jsonl_path=str(p))
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 6
    assert "winner_is_a" in json.loads(lines[0])
