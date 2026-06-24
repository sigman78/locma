import random

from locma.core import battle as b
from locma.core.actions import Pass
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.state import GameState
from locma.data.cards_db import load_cards


def _new_battle(seed=0):
    gs = GameState.new(random.Random(seed))
    b.start_battle(gs)
    return gs


def _drafted_battle(seed=1):
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards())
    for _ in range(60):
        apply_draft_pick(gs, 0)
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


def test_change_health_emits_face_damage_event():
    gs = _new_battle()
    gs.players[1].health = 30  # ensure non-fatal: 30 - 4 = 26 > 0
    events: list[dict] = []
    gs.emit = events.append
    b._change_health(gs, 1, 4, from_opponent=True)
    assert events == [{"t": "damage", "seat": 1, "target": "face", "amount": 4, "fatal": False}]


def test_change_health_healing_emits_nothing():
    gs = _new_battle()
    gs.players[0].health = 30
    events: list[dict] = []
    gs.emit = events.append
    b._change_health(gs, 0, -5)
    assert [e for e in events if e["t"] == "damage"] == []


def test_pass_decomposes_into_turn_events():
    gs = _drafted_battle()
    events: list[dict] = []
    gs.emit = events.append
    b.apply_battle(gs, Pass())  # turn-ending pass by seat 0
    tags = [e["t"] for e in events]
    assert tags[0] == "action_applied"
    assert "turn_ended" in tags and "turn_started" in tags
    ended = next(e for e in events if e["t"] == "turn_ended")
    started = next(e for e in events if e["t"] == "turn_started")
    assert ended["seat"] == 0
    assert started["seat"] == 1
    assert isinstance(started["draws"], list)
    # second player's first turn draws at least one card
    assert len(started["draws"]) >= 1
    assert events.index(ended) < events.index(started)
