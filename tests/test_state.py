import random

from locma.core.actions import Attack, Pass
from locma.core.state import GameState, PlayerState


def test_player_defaults():
    p = PlayerState()
    assert p.health == 30 and p.mana == 0 and p.damage_counter == 0
    assert p.deck == [] and p.hand == [] and p.board == []


def test_actions_face_sentinel():
    a = Attack(attacker_id=5, target_id=-1)
    assert a.target_id == -1
    assert isinstance(Pass(), Pass)


def test_clone_is_independent():
    gs = GameState.new(random.Random(0))
    gs2 = gs.clone()
    gs2.players[0].health = 1
    assert gs.players[0].health == 30
