from locma.core.actions import Attack, Pass, Summon, Use
from locma.core.engine import run_game
from locma.policies.battles import GreedyBattlePolicy
from locma.policies.composer import Composer
from locma.policies.drafts import GreedyDraftPolicy


def _greedy():
    return Composer(GreedyBattlePolicy(), GreedyDraftPolicy(), name="greedy")


def test_on_step_receives_draft_ints_then_battle_actions():
    steps = []
    run_game(_greedy(), _greedy(), seed=1, on_step=lambda s, a, gs: steps.append((s, a)))
    assert steps, "on_step should have been called"
    # draft picks are ints, battle actions are Action instances
    assert any(isinstance(a, int) for _, a in steps)
    assert any(isinstance(a, (Summon, Attack, Use, Pass)) for _, a in steps)
    # seats are always 0 or 1
    assert all(s in (0, 1) for s, _ in steps)


def test_on_step_none_is_default_and_harmless():
    r = run_game(_greedy(), _greedy(), seed=1)
    assert r.winner in (0, 1)
