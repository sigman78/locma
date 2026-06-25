from dataclasses import dataclass

from locma.core.actions import Attack, Pass, Summon
from locma.policies.battles import (
    GreedyBattlePolicy,
    GroundBattlePolicy,
    RandomBattlePolicy,
    ScriptedBattlePolicy,
)


@dataclass
class _C:
    instance_id: int
    attack: int


@dataclass
class _BV:
    op_health: int = 30
    my_board: tuple = ()
    my_hand: tuple = ()


def test_random_battle_reproducible_after_reset():
    p = RandomBattlePolicy("r", seed=2)
    legal = [Pass(), Summon(1), Attack(3, -1)]
    first = [p.battle_action(None, legal) for _ in range(5)]
    p.reset(2)
    assert [p.battle_action(None, legal) for _ in range(5)] == first


def test_battle_action_accepts_state_arg():
    # State is ignored by simple battle policies but must be accepted.
    assert GroundBattlePolicy("g").battle_action(_BV(), [Pass()], state="GS") == Pass()


def test_ground_prefers_face_attack():
    legal = [Summon(1), Attack(3, 7), Attack(3, -1)]
    assert GroundBattlePolicy("g").battle_action(_BV(), legal) == Attack(3, -1)


def test_greedy_lethal_swings_face():
    view = _BV(op_health=5, my_board=(_C(3, 5),))
    legal = [Attack(3, -1), Pass()]
    assert GreedyBattlePolicy("g").battle_action(view, legal) == Attack(3, -1)


def test_scripted_pass_when_only_pass():
    assert ScriptedBattlePolicy("s").battle_action(_BV(), [Pass()]) == Pass()
