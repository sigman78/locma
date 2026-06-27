"""Tactical observation extension for PPO experiments.

This keeps the production semantic action space unchanged, but appends two
feature groups to the base 308-d observation:

- visible static utility per card slot: player HP, enemy HP, card draw;
- shallow public tactical scalars derived from the current legal actions.

The goal is to test whether the reactive policy is missing cheap tactical facts
or card effects that are visible to a player but absent from the base vector.
"""

from __future__ import annotations

import numpy as np

from locma.core.actions import Attack, Summon, Use
from locma.core.cards import CardType
from locma.data.cards_db import load_cards
from locma.envs import encode as base

ACTION_SIZE = base.ACTION_SIZE
MAX_HAND = base.MAX_HAND
MAX_BOARD = base.MAX_BOARD
N_CARD_SLOTS = MAX_HAND + MAX_BOARD + MAX_BOARD
UTILITY_FEATS = 3
TACTICAL_FEATS = 24
OBS_SIZE = base.OBS_SIZE + N_CARD_SLOTS * UTILITY_FEATS + TACTICAL_FEATS

_CARDS_BY_ID = {c.id: c for c in load_cards()}


def _utility_block(c) -> list[float]:
    card = _CARDS_BY_ID.get(c.card_id)
    if card is None:
        return [0.0, 0.0, 0.0]
    return [float(card.player_hp), float(card.enemy_hp), float(card.card_draw)]


def _slot_utility_features(view) -> list[float]:
    out: list[float] = []
    for seq, n in (
        (view.my_hand, MAX_HAND),
        (view.my_board, MAX_BOARD),
        (view.op_board, MAX_BOARD),
    ):
        for i in range(n):
            out += _utility_block(seq[i]) if i < len(seq) else [0.0] * UTILITY_FEATS
    return out


def _by_id(seq) -> dict[int, object]:
    return {c.instance_id: c for c in seq}


def _dies_after_damage(card, damage: int, lethal: bool) -> bool:
    if damage <= 0:
        return False
    if "W" in card.abilities:
        return False
    return lethal or card.defense - damage <= 0


def _tactical_features(view, legal=None) -> list[float]:
    legal = legal or ()
    my_board = list(view.my_board)
    op_board = list(view.op_board)
    my_by_id = _by_id(my_board)
    op_by_id = _by_id(op_board)
    hand_by_id = _by_id(view.my_hand)

    guards = [c for c in op_board if "G" in c.abilities]
    ready = [c for c in my_board if c.can_attack and not c.has_attacked]

    face_attacks = [a for a in legal if isinstance(a, Attack) and a.target_id == -1]
    unit_attacks = [a for a in legal if isinstance(a, Attack) and a.target_id != -1]
    summons = [a for a in legal if isinstance(a, Summon)]
    uses = [a for a in legal if isinstance(a, Use)]

    reachable_face = sum(my_by_id[a.attacker_id].attack for a in face_attacks if a.attacker_id in my_by_id)
    face_blue_damage = 0
    green_uses = red_uses = blue_face_uses = blue_unit_uses = 0
    for a in uses:
        item = hand_by_id.get(a.item_instance_id)
        if item is None:
            continue
        if item.type == CardType.GREEN_ITEM:
            green_uses += 1
        elif item.type == CardType.RED_ITEM:
            red_uses += 1
        elif a.target_id == -1:
            blue_face_uses += 1
            face_blue_damage += max(0, -item.defense) + max(0, -_CARDS_BY_ID[item.card_id].enemy_hp)
        else:
            blue_unit_uses += 1

    best_trade_delta = 0.0
    favorable_trades = 0
    killing_attacks = 0
    for a in unit_attacks:
        atk = my_by_id.get(a.attacker_id)
        dfn = op_by_id.get(a.target_id)
        if atk is None or dfn is None:
            continue
        kills_defender = _dies_after_damage(dfn, atk.attack, "L" in atk.abilities)
        kills_attacker = _dies_after_damage(atk, dfn.attack, "L" in dfn.abilities)
        delta = 0.0
        if kills_defender:
            delta += dfn.attack + dfn.defense
            killing_attacks += 1
        if kills_attacker:
            delta -= atk.attack + atk.defense
        best_trade_delta = max(best_trade_delta, delta)
        favorable_trades += int(delta > 0)

    summon_cards = [hand_by_id[a.card_instance_id] for a in summons if a.card_instance_id in hand_by_id]
    best_summon_attack = max((c.attack for c in summon_cards), default=0)
    best_summon_defense = max((c.defense for c in summon_cards), default=0)
    cheapest_summon = min((c.cost for c in summon_cards), default=0)

    my_attack = sum(c.attack for c in my_board)
    op_attack = sum(c.attack for c in op_board)
    my_defense = sum(c.defense for c in my_board)
    op_defense = sum(c.defense for c in op_board)
    ready_attack = sum(c.attack for c in ready)
    opp_board_face = sum(c.attack for c in op_board if "G" not in c.abilities)
    exposed_to_lethal = float(opp_board_face >= view.me_health)
    lethal_available = float(reachable_face + face_blue_damage >= view.op_health)

    return [
        float(len(guards)),
        float(sum(c.defense for c in guards)),
        float(my_attack),
        float(op_attack),
        float(my_defense),
        float(op_defense),
        float(len(ready)),
        float(ready_attack),
        float(reachable_face),
        lethal_available,
        exposed_to_lethal,
        float(len(summons)),
        float(green_uses),
        float(red_uses),
        float(blue_face_uses),
        float(blue_unit_uses),
        float(len(face_attacks)),
        float(len(unit_attacks)),
        float(cheapest_summon),
        float(best_summon_attack),
        float(best_summon_defense),
        float(best_trade_delta),
        float(favorable_trades),
        float(killing_attacks),
    ]


def encode_battle(view, legal=None) -> np.ndarray:
    vec = list(base.encode_battle(view))
    vec += _slot_utility_features(view)
    vec += _tactical_features(view, legal)
    arr = np.asarray(vec, dtype=np.float32)
    assert len(arr) == OBS_SIZE, f"tactical encode length {len(arr)} != {OBS_SIZE}"
    return arr


sem_index = base.sem_index
action_mask = base.action_mask
index_to_action = base.index_to_action
