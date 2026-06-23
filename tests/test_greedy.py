from __future__ import annotations

from locma.core.views import CardView, DraftView
from locma.harness.match import run_match
from locma.policies.greedy import GreedyPolicy
from locma.policies.random_policy import RandomPolicy


def test_greedy_prefers_stronger_card():
    weak = CardView(-1, 1, 0, 2, 1, 1, "------")
    strong = CardView(-1, 2, 0, 2, 4, 4, "------")
    mid = CardView(-1, 3, 0, 2, 2, 2, "------")
    view = DraftView(0, (weak, strong, mid))
    assert GreedyPolicy("g").draft_action(view, [0, 1, 2]) == 1


def test_greedy_beats_random_over_many_games():
    res = run_match(GreedyPolicy("g"), RandomPolicy("r"), games=60, seed=0)
    assert res.win_rate_a > 0.5
