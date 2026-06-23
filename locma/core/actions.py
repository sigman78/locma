from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Summon:
    card_instance_id: int


@dataclass(frozen=True)
class Attack:
    attacker_id: int
    target_id: int  # -1 = face


@dataclass(frozen=True)
class Use:
    item_instance_id: int
    target_id: int  # -1 = face/no target


@dataclass(frozen=True)
class Pass:
    pass


Action = Summon | Attack | Use | Pass
