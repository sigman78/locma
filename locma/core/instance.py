from __future__ import annotations
from dataclasses import dataclass
from locma.core.cards import Card, ABILITY_ORDER


@dataclass
class CardInstance:
    card: Card
    instance_id: int
    attack: int
    defense: int
    abilities: str
    can_attack: bool = False
    has_attacked: bool = False

    @classmethod
    def from_card(cls, card: Card, instance_id: int) -> CardInstance:
        return cls(card=card, instance_id=instance_id,
                   attack=card.attack, defense=card.defense,
                   abilities=card.abilities,
                   can_attack=card.has("C"))

    def has(self, ability: str) -> bool:
        return self.abilities[ABILITY_ORDER.index(ability)] != "-"
