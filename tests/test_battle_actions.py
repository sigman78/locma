from __future__ import annotations

import random

from locma.core.actions import Attack, Pass, Summon, Use
from locma.core.battle import apply_battle, battle_legal
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase


def _bare_battle():
    gs = GameState.new(random.Random(0))
    gs.phase = Phase.BATTLE
    gs.turn = 1
    gs.current = 0
    gs.players[0].mana = 5
    gs.players[0].max_mana = 5
    return gs


def _creature(iid, cost=2, atk=3, dfn=2, ab=""):
    c = Card(100, "X", CardType.CREATURE, cost, atk, dfn, normalize_abilities(ab), 0, 0, 0)
    return CardInstance.from_card(c, iid)


def _blue_item(iid, *, defense=0, player_hp=0, enemy_hp=0, cost=2, card_draw=0):
    c = Card(
        200,
        "Blue",
        CardType.BLUE_ITEM,
        cost,
        0,
        defense,
        normalize_abilities(""),
        player_hp,
        enemy_hp,
        card_draw,
    )
    return CardInstance.from_card(c, iid)


def _red_item(iid, *, defense=-1, cost=2, card_draw=0, ab=""):
    c = Card(
        201,
        "Red",
        CardType.RED_ITEM,
        cost,
        0,
        defense,
        normalize_abilities(ab),
        0,
        0,
        card_draw,
    )
    return CardInstance.from_card(c, iid)


def _green_item(iid, *, defense=3, cost=2, card_draw=0, ab=""):
    c = Card(
        202,
        "Green",
        CardType.GREEN_ITEM,
        cost,
        0,
        defense,
        normalize_abilities(ab),
        0,
        0,
        card_draw,
    )
    return CardInstance.from_card(c, iid)


def _use_targets(legal, iid):
    return {a.target_id for a in legal if isinstance(a, Use) and a.item_instance_id == iid}


def test_summon_moves_card_to_board_and_spends_mana():
    gs = _bare_battle()
    inst = _creature(1, cost=3)
    gs.players[0].hand.append(inst)
    legal = battle_legal(gs)
    assert any(isinstance(a, Summon) and a.card_instance_id == 1 for a in legal)
    apply_battle(gs, Summon(1))
    assert gs.players[0].mana == 2
    assert len(gs.players[0].board) == 1 and not gs.players[0].hand


def test_pass_ends_turn():
    gs = _bare_battle()
    apply_battle(gs, Pass())
    assert gs.current == 1


def test_guard_restricts_attack_targets():
    gs = _bare_battle()
    atk = _creature(1)
    atk.can_attack = True
    gs.players[0].board.append(atk)
    guard = _creature(2, ab="G")
    plain = _creature(3)
    gs.players[1].board.extend([guard, plain])
    targets = [
        a.target_id for a in battle_legal(gs) if isinstance(a, Attack) and a.attacker_id == 1
    ]
    assert 2 in targets and 3 not in targets and -1 not in targets


def test_zero_attack_creature_cannot_attack():
    # House rule (stricter than the LOCM 1.5 referee): a creature with 0 attack
    # has no legal Attack actions -- neither face nor creatures.
    gs = _bare_battle()
    wall = _creature(1, atk=0, dfn=5)
    wall.can_attack = True
    gs.players[0].board.append(wall)
    gs.players[1].board.append(_creature(2))
    assert not any(isinstance(a, Attack) and a.attacker_id == 1 for a in battle_legal(gs))


def test_zero_attack_gate_follows_current_attack():
    # The gate reads the instance's CURRENT attack: a green buff mid-turn
    # re-enables attacking, a red debuff to 0 disables it.
    gs = _bare_battle()
    wall = _creature(1, atk=0, dfn=5)
    wall.can_attack = True
    gs.players[0].board.append(wall)
    assert not any(isinstance(a, Attack) and a.attacker_id == 1 for a in battle_legal(gs))
    wall.attack = 2  # green-item buff
    assert any(isinstance(a, Attack) and a.attacker_id == 1 for a in battle_legal(gs))
    wall.attack = 0  # red-item debuff back to 0
    assert not any(isinstance(a, Attack) and a.attacker_id == 1 for a in battle_legal(gs))


def test_blue_item_zero_defense_only_targets_face():
    # Healing Potion (defense 0, +5 self-heal): per the rules a blue item may
    # target an enemy creature only with NEGATIVE defense. With defense 0 the
    # only legal target is -1 (no-creature / face), even when enemies are present.
    gs = _bare_battle()
    gs.players[1].board.extend([_creature(10), _creature(11)])
    heal = _blue_item(1, defense=0, player_hp=5)
    gs.players[0].hand.append(heal)
    assert _use_targets(battle_legal(gs), 1) == {-1}


def test_blue_lifesteal_zero_defense_only_targets_face():
    # Minor Life Steal (defense 0, enemy_hp -2, player_hp +2): its damage is
    # opponent-face damage carried by enemy_hp, not by defense, so it must not
    # be offered against enemy creatures.
    gs = _bare_battle()
    gs.players[1].board.extend([_creature(10), _creature(11)])
    steal = _blue_item(1, defense=0, player_hp=2, enemy_hp=-2)
    gs.players[0].hand.append(steal)
    assert _use_targets(battle_legal(gs), 1) == {-1}


def test_blue_item_negative_defense_targets_enemy_creatures_and_face():
    # Tome of Thunder-style blue (defense -4): may hit any enemy creature OR -1.
    gs = _bare_battle()
    gs.players[1].board.extend([_creature(10), _creature(11)])
    bolt = _blue_item(1, defense=-4)
    gs.players[0].hand.append(bolt)
    assert _use_targets(battle_legal(gs), 1) == {10, 11, -1}


# --- card_draw on a played card grants a (deferred, next-turn) bonus draw -----
# Faithful to gym-locm's _do_use / _do_summon: `bonus_draw += card_draw`, applied
# for EVERY card type. The draw itself lands at the start of the actor's next turn
# (consumed in start_turn). Here we assert the bonus_draw accrual at play time.


def test_summon_card_draw_grants_bonus_draw():
    # Regression guard: creatures already accrue it (e.g. Eldritch Cyclops).
    gs = _bare_battle()
    cyclops = _creature(1, cost=3)
    cyclops.card = Card(100, "Drawer", CardType.CREATURE, 3, 3, 5, normalize_abilities(""), 0, 0, 1)
    gs.players[0].hand.append(cyclops)
    apply_battle(gs, Summon(1))
    assert gs.players[0].bonus_draw == 1


def test_green_item_card_draw_grants_bonus_draw():
    # Regression guard: green items already accrue it (e.g. Enchanted Hat).
    gs = _bare_battle()
    gs.players[0].board.append(_creature(10))
    item = _green_item(1, card_draw=1)
    gs.players[0].hand.append(item)
    apply_battle(gs, Use(1, 10))
    assert gs.players[0].bonus_draw == 1


def test_red_item_card_draw_grants_bonus_draw():
    # Quick Shot / Mighty Throwing Axe: "Deal N damage to an enemy creature. Draw a card."
    gs = _bare_battle()
    gs.players[1].board.append(_creature(10))
    item = _red_item(1, defense=-1, card_draw=1)
    gs.players[0].hand.append(item)
    apply_battle(gs, Use(1, 10))
    assert gs.players[0].bonus_draw == 1


def test_blue_item_card_draw_grants_bonus_draw():
    # Poison: "Deal 2 damage to your opponent. Draw a card." (face-targeted blue).
    gs = _bare_battle()
    item = _blue_item(1, enemy_hp=-2, card_draw=1)
    gs.players[0].hand.append(item)
    apply_battle(gs, Use(1, -1))
    assert gs.players[0].bonus_draw == 1
