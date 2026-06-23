from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy
from locma.core.actions import Pass, Summon

def test_random_draft_in_range():
    p = RandomPolicy("r", seed=1)
    assert p.draft_action(None, [0, 1, 2]) in (0, 1, 2)

def test_random_battle_returns_legal():
    p = RandomPolicy("r", seed=1)
    legal = [Pass()]
    assert p.battle_action(None, legal) in legal

def test_scripted_draft_picks_zero():
    assert ScriptedPolicy("s").draft_action(None, [0, 1, 2]) == 0

def test_scripted_battle_skips_pass():
    legal = [Pass(), Summon(1)]
    assert ScriptedPolicy("s").battle_action(None, legal) == Summon(1)

def test_scripted_battle_all_pass_returns_pass():
    result = ScriptedPolicy("s").battle_action(None, [Pass()])
    assert isinstance(result, Pass)
