from __future__ import annotations

from typing import Protocol


class Policy(Protocol):
    name: str

    def draft_action(self, view, legal: list[int]) -> int: ...
    def battle_action(self, view, legal: list): ...
    def reset(self, seed: int | None = None) -> None: ...


class CompositePolicy:
    def __init__(self, draft, battle, name=None):
        self.draft = draft
        self.battle = battle
        self.name = name or f"{draft.name}+{battle.name}"

    def draft_action(self, view, legal):
        return self.draft.draft_action(view, legal)

    def battle_action(self, view, legal):
        return self.battle.battle_action(view, legal)

    def reset(self, seed=None):
        self.draft.reset(seed)
        self.battle.reset(seed)
