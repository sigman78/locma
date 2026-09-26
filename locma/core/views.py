from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CardView:
    instance_id: int
    card_id: int
    type: int
    cost: int
    attack: int
    defense: int
    abilities: str
    can_attack: bool = False
    has_attacked: bool = False


@dataclass(frozen=True, slots=True)
class DraftView:
    round: int
    offered: tuple  # 3 CardView (instance_id unused, -1)
    taken: int | None = None  # shared draft: index the first picker took this round


@dataclass(frozen=True, slots=True)
class BattleView:
    turn: int
    me_health: int
    me_mana: int
    op_health: int
    op_hand_count: int
    my_hand: tuple
    my_board: tuple
    op_board: tuple
    my_deck_count: int = 0
    op_deck_count: int = 0
    my_deck_cards: tuple = ()  # SORTED remaining own card ids (order-destroyed multiset)
    op_played: tuple = ()  # opponent's played card ids, in play order
    op_turn_log: tuple = ()  # opponent's (mana_left, hand_size) at each of their turn ends
