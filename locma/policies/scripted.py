from __future__ import annotations

from locma.core.actions import Pass


class ScriptedPolicy:
    def __init__(self, name: str = "scripted"):
        self.name = name

    def draft_action(self, view, legal):
        return legal[0]

    def battle_action(self, view, legal):
        for a in legal:
            if not isinstance(a, Pass):
                return a
        return Pass()

    def reset(self, seed=None):
        pass
