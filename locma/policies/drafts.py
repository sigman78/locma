from __future__ import annotations

import random

_CREATURE = 0  # CardView.type for creatures


class RandomDraftPolicy:
    def __init__(self, name: str = "random-draft", seed: int = 0):
        self.name = name
        self._seed = seed
        self._r = random.Random(seed)

    def draft_action(self, view, legal):
        return self._r.choice(legal)

    def reset(self, seed=None):
        self._r = random.Random(self._seed if seed is None else seed)


def _kw_count(abilities: str) -> int:
    return sum(1 for ch in abilities if ch != "-")


def _score(cv) -> float:
    base = cv.attack + cv.defense + 0.5 * _kw_count(cv.abilities)
    if cv.type != 0:  # items slightly deprioritized in draft
        base -= 1.0
    return base


class GreedyDraftPolicy:
    def __init__(self, name: str = "greedy-draft"):
        self.name = name

    def draft_action(self, view, legal):
        scores = [_score(cv) for cv in view.offered]
        return max(legal, key=lambda i: scores[i])

    def reset(self, seed=None):
        pass


class MaxGuardDraftPolicy:
    """Draft Guard creatures above all else."""

    def __init__(self, name: str = "max-guard-draft"):
        self.name = name

    def draft_action(self, view, legal):
        def key(i):
            cv = view.offered[i]
            is_creature = cv.type == _CREATURE
            has_guard = is_creature and "G" in cv.abilities
            return (has_guard, is_creature, cv.attack + cv.defense)

        return max(legal, key=key)

    def reset(self, seed=None):
        pass


class MaxAttackDraftPolicy:
    """Draft the highest-attack creature."""

    def __init__(self, name: str = "max-attack-draft"):
        self.name = name

    def draft_action(self, view, legal):
        def key(i):
            cv = view.offered[i]
            is_creature = cv.type == _CREATURE
            return (is_creature, cv.attack, cv.defense)

        return max(legal, key=key)

    def reset(self, seed=None):
        pass
