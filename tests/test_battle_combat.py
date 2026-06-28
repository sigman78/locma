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


def _red(iid, atk, dfn, ab=""):
    return CardInstance.from_card(
        Card(3, "Red", CardType.RED_ITEM, 1, atk, dfn, normalize_abilities(ab), 0, 0, 0), iid
    )


def _blue(iid, dfn, ab=""):
    return CardInstance.from_card(
        Card(4, "Blue", CardType.BLUE_ITEM, 1, 0, dfn, normalize_abilities(ab), 0, 0, 0), iid
    )


def test_red_item_damage_absorbed_by_ward():
    """A red item's damage (negative defense) on a warded creature is fully soaked;
    Ward is consumed and the creature keeps all its defense (matches combat)."""
    gs = _gs()
    gs.players[0].mana = 10
    warded = _c(2, 1, 3, ab="W")
    gs.players[1].board.append(warded)
    gs.players[0].hand.append(_red(5, 0, -2))  # -2 damage
    apply_battle(gs, Use(5, 2))
    assert warded.defense == 3  # ward soaked all the damage
    assert warded.has("W") is False  # ward consumed
    assert warded in gs.players[1].board


def test_blue_item_damage_absorbed_by_ward():
    """A blue item's creature damage (negative defense) is likewise soaked by Ward."""
    gs = _gs()
    gs.players[0].mana = 10
    warded = _c(2, 1, 4, ab="W")
    gs.players[1].board.append(warded)
    gs.players[0].hand.append(_blue(5, -4))  # Tome-of-Thunder-like: -4 to a creature
    apply_battle(gs, Use(5, 2))
    assert warded.defense == 4
    assert warded.has("W") is False
    assert warded in gs.players[1].board


def test_red_item_that_strips_ward_then_damages():
    """Keyword removal runs BEFORE the damage: a red item that removes Ward (e.g.
    Decimate) strips it first, so the damage then lands in full."""
    gs = _gs()
    gs.players[0].mana = 10
    warded = _c(2, 1, 3, ab="W")
    gs.players[1].board.append(warded)
    gs.players[0].hand.append(_red(5, 0, -2, ab="BCDGLW"))  # removes all keywords + 2 damage
    apply_battle(gs, Use(5, 2))
    assert warded.has("W") is False  # stripped by the keyword removal
    assert warded.defense == 1  # ...so the 2 damage then landed (3 -> 1)


def test_zero_damage_red_item_does_not_consume_ward():
    """A red item that deals no damage (defense 0, e.g. Pierce Armour) must NOT
    consume Ward — only actual damage trips it."""
    gs = _gs()
    gs.players[0].mana = 10
    warded = _c(2, 1, 3, ab="W")
    gs.players[1].board.append(warded)
    gs.players[0].hand.append(_red(5, 0, 0, ab="---G--"))  # removes Guard only, no damage
    apply_battle(gs, Use(5, 2))
    assert warded.has("W") is True  # ward untouched
    assert warded.defense == 3
