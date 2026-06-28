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


def _blue_item(iid, *, defense=0, player_hp=0, enemy_hp=0, cost=2):
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
        0,
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
