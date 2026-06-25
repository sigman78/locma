import random

import pytest

from locma.core import battle as b
from locma.core.actions import Attack, Pass
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards


def _gs():
    gs = GameState.new(random.Random(0))
    gs.phase = Phase.BATTLE
    gs.current = 0
    return gs


def _c(iid, atk, dfn, ab=""):
    card = Card(1, "X", CardType.CREATURE, 1, atk, dfn, normalize_abilities(ab), 0, 0, 0)
    inst = CardInstance.from_card(card, iid)
    inst.can_attack = True
    return inst


@pytest.fixture
def make_combat_gs():
    def _make(attacker_atk, defender_def):
        gs = _gs()
        atk_unit = _c(1, attacker_atk, 10)
        dfn_unit = _c(7, 0, defender_def)
        dfn_unit.can_attack = True
        dfn_unit.has_attacked = False
        gs.players[0].board.append(atk_unit)
        gs.players[1].board.append(dfn_unit)
        return gs

    return _make


def _events_for(gs, action):
    events: list[dict] = []
    b.apply_battle(gs, action, emit=events.append)
    return events


def test_lethal_attack_emits_damage_and_unit_died(make_combat_gs):
    # make_combat_gs: a fixture/helper that returns a battle gs where seat 0 has
    # attacker iid=1 (attack 5) and seat 1 has defender iid=7 (defense 3, no W/L).
    gs = make_combat_gs(attacker_atk=5, defender_def=3)
    events = _events_for(gs, Attack(attacker_id=1, target_id=7))
    dmg = [e for e in events if e["t"] == "damage" and e["target"] == 7]
    assert dmg and dmg[0]["seat"] == 1 and dmg[0]["fatal"] is True
    assert dmg[0]["amount"] == 5  # equals attacker's attack
    died = [e for e in events if e["t"] == "unit_died" and e["iid"] == 7]
    assert died and died[0]["seat"] == 1


def test_nonlethal_attack_emits_damage_not_died(make_combat_gs):
    gs = make_combat_gs(attacker_atk=2, defender_def=5)
    events = _events_for(gs, Attack(attacker_id=1, target_id=7))
    dmg = [e for e in events if e["t"] == "damage" and e["target"] == 7]
    assert dmg and dmg[0]["amount"] == 2 and dmg[0]["fatal"] is False
    assert not [e for e in events if e["t"] == "unit_died"]


def test_attacker_retaliation_emits_damage_for_both():
    # Defender has nonzero attack and both survive, so the attacker takes
    # retaliation damage -- exercises the atk_applied emit block (seat 0 = cur).
    gs = _gs()
    atk_unit = _c(1, 1, 10)
    dfn_unit = _c(7, 3, 10)
    gs.players[0].board.append(atk_unit)
    gs.players[1].board.append(dfn_unit)
    events = _events_for(gs, Attack(attacker_id=1, target_id=7))
    assert {
        "t": "damage",
        "seat": 0,
        "target": 1,
        "amount": 3,
        "fatal": False,
    } in events
    assert {
        "t": "damage",
        "seat": 1,
        "target": 7,
        "amount": 1,
        "fatal": False,
    } in events
    assert not [e for e in events if e["t"] == "unit_died"]


def _item(iid, item_type, atk=0, dfn=0, ab=""):
    card = Card(100, "I", item_type, 1, atk, dfn, normalize_abilities(ab), 0, 0, 0)
    return CardInstance.from_card(card, iid)


def test_red_item_lethal_emits_damage_and_unit_died():
    # RED_ITEM with negative defense reduces the opponent unit to <=0:
    # emits both a fatal damage event and a unit_died for the opponent's unit.
    gs = _gs()
    tgt = _c(7, 0, 3)
    gs.players[1].board.append(tgt)
    item = _item(2, CardType.RED_ITEM, dfn=-5)
    events: list[dict] = []
    b._apply_item(gs, item, 7, emit=events.append)
    dmg = [e for e in events if e["t"] == "damage" and e["target"] == 7]
    assert dmg and dmg[0]["seat"] == 1 and dmg[0]["fatal"] is True
    assert dmg[0]["amount"] == 5  # before(3) - after(-2) defense delta
    died = [e for e in events if e["t"] == "unit_died" and e["iid"] == 7]
    assert died and died[0]["seat"] == 1


def _new_battle(seed=0):
    gs = GameState.new(random.Random(seed))
    b.start_battle(gs)
    return gs


def _drafted_battle(seed=1):
    # Install the emit sink AFTER this returns (e.g. pass emit= to apply_battle),
    # not into start_battle here — otherwise you'd also capture turn 1's
    # start-of-turn turn_started event emitted during start_battle.
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards())
    for _ in range(60):
        apply_draft_pick(gs, 0)
    b.start_battle(gs)
    return gs


def test_emit_sink_collects_events():
    gs = _new_battle()
    events: list[dict] = []
    b.apply_battle(gs, Pass(), emit=events.append)
    # A turn-ending Pass decomposes into turn events; the actor's turn ends first.
    assert events[0] == {"t": "turn_ended", "seat": 0}


def test_emit_is_noop_when_sink_none():
    gs = _new_battle()
    b.apply_battle(gs, Pass())  # must not raise


def test_change_health_emits_face_damage_event():
    gs = _new_battle()
    gs.players[1].health = 30  # ensure non-fatal: 30 - 4 = 26 > 0
    events: list[dict] = []
    b._change_health(gs, 1, 4, from_opponent=True, emit=events.append)
    assert events == [{"t": "damage", "seat": 1, "target": "face", "amount": 4, "fatal": False}]


def test_change_health_healing_emits_nothing():
    gs = _new_battle()
    gs.players[0].health = 30
    events: list[dict] = []
    b._change_health(gs, 0, -5, emit=events.append)
    assert [e for e in events if e["t"] == "damage"] == []


def test_pass_decomposes_into_turn_events():
    gs = _drafted_battle()
    events: list[dict] = []
    b.apply_battle(gs, Pass(), emit=events.append)  # turn-ending pass by seat 0
    tags = [e["t"] for e in events]
    assert tags[0] == "turn_ended"
    assert "turn_ended" in tags and "turn_started" in tags
    ended = next(e for e in events if e["t"] == "turn_ended")
    started = next(e for e in events if e["t"] == "turn_started")
    assert ended["seat"] == 0
    assert started["seat"] == 1
    assert isinstance(started["draws"], list)
    # second player's first turn draws at least one card
    assert len(started["draws"]) >= 1
    assert events.index(ended) < events.index(started)
