from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Summon:
    card_instance_id: int


@dataclass(frozen=True, slots=True)
class Attack:
    attacker_id: int
    target_id: int  # -1 = face


@dataclass(frozen=True, slots=True)
class Use:
    item_instance_id: int
    target_id: int  # -1 = face/no target


@dataclass(frozen=True, slots=True)
class Pass:
    pass


Action = Summon | Attack | Use | Pass


def action_to_dict(action: Action) -> dict:
    match action:
        case Summon(card_instance_id=iid):
            return {"t": "summon", "id": iid}
        case Attack(attacker_id=aid, target_id=tid):
            return {"t": "attack", "a": aid, "target": tid}
        case Use(item_instance_id=iid, target_id=tid):
            return {"t": "use", "item": iid, "target": tid}
        case Pass():
            return {"t": "pass"}
        case _:
            raise TypeError(f"unknown action: {action!r}")


def action_from_dict(d: dict) -> Action:
    match d["t"]:
        case "summon":
            return Summon(d["id"])
        case "attack":
            return Attack(d["a"], d["target"])
        case "use":
            return Use(d["item"], d["target"])
        case "pass":
            return Pass()
        case _:
            raise ValueError(f"unknown action tag: {d['t']!r}")
