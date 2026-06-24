import random

from locma.core import battle as b
from locma.core.actions import Pass
from locma.core.state import GameState


def _new_battle(seed=0):
    gs = GameState.new(random.Random(seed))
    b.start_battle(gs)
    return gs


def test_emit_sink_collects_action_applied():
    gs = _new_battle()
    events: list[dict] = []
    gs.emit = events.append
    b.apply_battle(gs, Pass())
    assert events[0]["t"] == "action_applied"
    assert events[0]["seat"] == 0
    assert events[0]["action"] == {"t": "pass"}


def test_emit_is_noop_when_sink_none():
    gs = _new_battle()
    assert gs.emit is None
    b.apply_battle(gs, Pass())  # must not raise
