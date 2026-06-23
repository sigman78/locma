from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CardView:
    instance_id: int
    card_id: int
    type: int
    cost: int
    attack: int
    defense: int
    abilities: str


@dataclass(frozen=True)
class DraftView:
    round: int
    offered: tuple  # 3 CardView (instance_id unused, -1)


@dataclass(frozen=True)
class BattleView:
    turn: int
    me_health: int
    me_mana: int
    op_health: int
    op_hand_count: int
    my_hand: tuple
    my_board: tuple
    op_board: tuple
