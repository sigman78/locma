from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

ABILITY_ORDER = "BCDGLW"  # Breakthrough, Charge, Drain, Guard, Lethal, Ward


class CardType(IntEnum):
    CREATURE = 0
    GREEN_ITEM = 1
    RED_ITEM = 2
    BLUE_ITEM = 3


def normalize_abilities(raw: str) -> str:
    present = {ch for ch in raw if ch in ABILITY_ORDER}
    return "".join(ch if ch in present else "-" for ch in ABILITY_ORDER)


@dataclass(frozen=True)
class Card:
    id: int
    name: str
    type: CardType
    cost: int
    attack: int
    defense: int
    abilities: str  # length-6 mask over ABILITY_ORDER
    player_hp: int
    enemy_hp: int
    card_draw: int

    def has(self, ability: str) -> bool:
        return self.abilities[ABILITY_ORDER.index(ability)] != "-"
