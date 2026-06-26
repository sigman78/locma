import random

from locma.core.actions import Attack, Use
from locma.core.battle import _resolve_attack, apply_battle, battle_legal
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase


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


def test_trade_kills_both():
    gs = _gs()
    a = _c(1, 3, 2)
    d = _c(2, 2, 2)
    gs.players[0].board.append(a)
    gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert gs.players[0].board == [] and gs.players[1].board == []


def test_ward_absorbs_then_drops():
    gs = _gs()
    a = _c(1, 3, 2)
    d = _c(2, 1, 3, ab="W")
    gs.players[0].board.append(a)
    gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert d.defense == 3 and d.has("W") is False  # warded, ward consumed
    assert a.defense == 1  # attacker still took 1


def test_lethal_destroys_big():
    gs = _gs()
    a = _c(1, 1, 5, ab="L")
    d = _c(2, 0, 9)
    gs.players[0].board.append(a)
    gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert gs.players[1].board == []


def test_breakthrough_tramples_face():
    gs = _gs()
    a = _c(1, 5, 5, ab="B")
    d = _c(2, 0, 2)
    gs.players[0].board.append(a)
    gs.players[1].board.append(d)
    _resolve_attack(gs, 1, 2)
    assert gs.players[1].health == 30 - 3  # 5 - 2 defense = 3 trample


def test_drain_heals_on_face():
    gs = _gs()
    gs.players[0].health = 20
    a = _c(1, 4, 2, ab="D")
    gs.players[0].board.append(a)
    _resolve_attack(gs, 1, -1)
    assert gs.players[1].health == 30 - 4 and gs.players[0].health == 24


def _sick_creature(iid):
    """A vanilla creature placed on the board this turn: summoning-sick (no Charge)."""
    inst = CardInstance.from_card(
        Card(1, "Sleepy", CardType.CREATURE, 1, 3, 3, normalize_abilities(""), 0, 0, 0), iid
    )
    assert inst.can_attack is False  # from_card: can_attack = has Charge
    return inst


def _green(iid, atk, dfn, ab):
    return CardInstance.from_card(
        Card(2, "Item", CardType.GREEN_ITEM, 1, atk, dfn, normalize_abilities(ab), 0, 0, 0), iid
    )


def test_green_item_granting_charge_wakes_summoning_sick_creature():
    """Casting a green Charge item on a just-summoned (summoning-sick) creature lets
    it attack this turn — gaining Charge removes summoning sickness (LOCM 1.5)."""
    gs = _gs()
    p = gs.players[0]
    p.mana = 10
    sick = _sick_creature(1)
    p.board.append(sick)
    p.hand.append(_green(2, 0, 0, "C"))  # "Grow Wings": Charge only
    apply_battle(gs, Use(2, 1))  # cast the Charge item on the sleeping creature
    assert sick.can_attack is True  # it woke up
    assert any(isinstance(a, Attack) and a.attacker_id == 1 for a in battle_legal(gs))


def test_green_item_without_charge_leaves_creature_summoning_sick():
    """A stat-only green item must NOT wake a summoning-sick creature."""
    gs = _gs()
    p = gs.players[0]
    p.mana = 10
    sick = _sick_creature(1)
    p.board.append(sick)
    p.hand.append(_green(2, 2, 2, ""))  # pure stat buff, no Charge
    apply_battle(gs, Use(2, 1))
    assert sick.can_attack is False  # still summoning sick
    assert not any(isinstance(a, Attack) and a.attacker_id == 1 for a in battle_legal(gs))


def test_charge_item_does_not_grant_a_second_attack():
    """Granting Charge to a creature that already swung does NOT let it attack again
    — has_attacked still guards the double-swing."""
    gs = _gs()
    p = gs.players[0]
    p.mana = 10
    a = _c(1, 3, 3)  # ready creature (can_attack True)
    p.board.append(a)
    gs.players[1].board.append(_c(2, 1, 1))
    _resolve_attack(gs, 1, -1)  # it attacks face -> has_attacked True, can_attack False
    assert a.has_attacked is True and a.can_attack is False
    p.hand.append(_green(3, 0, 0, "C"))
    apply_battle(gs, Use(3, 1))  # grant Charge after it already swung
    assert not any(isinstance(act, Attack) and act.attacker_id == 1 for act in battle_legal(gs))
