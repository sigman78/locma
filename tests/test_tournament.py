from __future__ import annotations

from locma.harness.tournament import run_tournament
from locma.policies.battles import RandomBattlePolicy, ScriptedBattlePolicy
from locma.policies.composer import Composer
from locma.policies.drafts import RandomDraftPolicy
from locma.stats.ratings import elo_update


def _random(name):
    return Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name=name)


def test_elo_winner_gains():
    ra, rb = elo_update(1500, 1500, 1.0)
    assert ra > 1500 > rb


def test_tournament_structure():
    pols = [
        _random("r"),
        Composer(ScriptedBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name="s"),
    ]
    res = run_tournament(pols, games=6, seed=0, reference="r")
    assert ("r", "s") in res.win_matrix
    assert set(res.ratings) == {"r", "s"}
    assert "s" in res.p_vs_reference
