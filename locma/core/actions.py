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


def action_to_dict(action: Action) -> dict:
    if isinstance(action, Summon):
        return {"t": "summon", "id": action.card_instance_id}
    if isinstance(action, Attack):
        return {"t": "attack", "a": action.attacker_id, "target": action.target_id}
    if isinstance(action, Use):
        return {"t": "use", "item": action.item_instance_id, "target": action.target_id}
    if isinstance(action, Pass):
        return {"t": "pass"}
    raise TypeError(f"unknown action: {action!r}")


def action_from_dict(d: dict) -> Action:
    t = d["t"]
    if t == "summon":
        return Summon(d["id"])
    if t == "attack":
        return Attack(d["a"], d["target"])
    if t == "use":
        return Use(d["item"], d["target"])
    if t == "pass":
        return Pass()
    raise ValueError(f"unknown action tag: {t!r}")
