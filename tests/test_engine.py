from __future__ import annotations

from locma.core.engine import run_game
from locma.policies.battles import RandomBattlePolicy
from locma.policies.composer import Composer
from locma.policies.drafts import RandomDraftPolicy


def _random(name):
    return Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name=name)


def test_run_game_returns_winner():
    r = run_game(_random("a"), _random("b"), seed=123)
    assert r.winner in (0, 1)
    assert r.turns >= 1


def test_run_game_is_deterministic():
    r1 = run_game(_random("a"), _random("b"), seed=999)
    r2 = run_game(_random("a"), _random("b"), seed=999)
    assert (r1.winner, r1.turns) == (r2.winner, r2.turns)
