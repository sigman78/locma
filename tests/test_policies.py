from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy
from locma.core.actions import Pass

def test_random_draft_in_range():
    p = RandomPolicy("r", seed=1)
    assert p.draft_action(None, [0, 1, 2]) in (0, 1, 2)

def test_random_battle_returns_legal():
    p = RandomPolicy("r", seed=1)
    legal = [Pass()]
    assert p.battle_action(None, legal) in legal

def test_scripted_draft_picks_zero():
    assert ScriptedPolicy("s").draft_action(None, [0, 1, 2]) == 0
