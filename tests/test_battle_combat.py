import random
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.core.battle import _resolve_attack

def _gs():
    gs = GameState.new(random.Random(0)); gs.phase = Phase.BATTLE; gs.current = 0
    return gs

def _c(iid, atk, dfn, ab=""):
    card = Card(1, "X", CardType.CREATURE, 1, atk, dfn, normalize_abilities(ab), 0, 0, 0)
    inst = CardInstance.from_card(card, iid); inst.can_attack = True
    return inst

def test_trade_kills_both():
    gs = _gs()
    a = _c(1, 3, 2); d = _c(2, 2, 2)
    gs.players[0].board.append(a); gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert gs.players[0].board == [] and gs.players[1].board == []

def test_ward_absorbs_then_drops():
    gs = _gs()
    a = _c(1, 3, 2); d = _c(2, 1, 3, ab="W")
    gs.players[0].board.append(a); gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert d.defense == 3 and d.has("W") is False  # warded, ward consumed
    assert a.defense == 1                            # attacker still took 1

def test_lethal_destroys_big():
    gs = _gs()
    a = _c(1, 1, 5, ab="L"); d = _c(2, 0, 9)
    gs.players[0].board.append(a); gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert gs.players[1].board == []

def test_breakthrough_tramples_face():
    gs = _gs()
    a = _c(1, 5, 5, ab="B"); d = _c(2, 0, 2)
    gs.players[0].board.append(a); gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert gs.players[1].health == 30 - 3   # 5 - 2 defense = 3 trample

def test_drain_heals_on_face():
    gs = _gs(); gs.players[0].health = 20
    a = _c(1, 4, 2, ab="D")
    gs.players[0].board.append(a)
    _resolve_attack(gs, 1, -1)
    assert gs.players[1].health == 30 - 4 and gs.players[0].health == 24
