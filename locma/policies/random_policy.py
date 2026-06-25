from __future__ import annotations

import random


class RandomPolicy:
    def __init__(self, name: str = "random", seed: int = 0):
        self.name = name
        self._seed = seed
        self._r = random.Random(seed)

    def draft_action(self, view, legal):
        return self._r.choice(legal)

    def battle_action(self, view, legal, state=None):
        return self._r.choice(legal)

    def reset(self, seed=None):
        self._r = random.Random(self._seed if seed is None else seed)
