from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DraftPolicy(Protocol):
    name: str

    def draft_action(self, view, legal: list[int]) -> int: ...
    def reset(self, seed: int | None = None) -> None: ...


@runtime_checkable
class BattlePolicy(Protocol):
    name: str

    def battle_action(self, view, legal: list, state=None): ...
    def reset(self, seed: int | None = None) -> None: ...


@runtime_checkable
class Policy(Protocol):
    """The engine-facing contract: answers every phase the engine asks of a seat."""

    name: str

    def draft_action(self, view, legal: list[int]) -> int: ...
    def battle_action(self, view, legal: list, state=None): ...
    def reset(self, seed: int | None = None) -> None: ...
