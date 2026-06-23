from __future__ import annotations

import random

from locma.core.actions import Attack, Pass, Summon
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
