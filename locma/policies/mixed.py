"""Mixed-opponent policy: a training opponent that varies per episode.

`MixedOpponentPolicy` is a full Policy (draft + battle) that delegates each
*episode* to a different sub-policy sampled from a pool, giving an RL learner a
varied opponent distribution instead of a single fixed style.

A new episode is detected from the draft round rolling *back* to a lower value
(the draft runs round 0..29 within an episode, then a fresh episode restarts at
0), or on the first draft action after construction/`reset`. The chosen
sub-policy then drives both draft and battle for that whole episode. This needs
no cooperation from `BattleEnv` (which does not reset its opponent per episode),
so single-opponent training and replay are unaffected.
"""

from __future__ import annotations

import random


class MixedOpponentPolicy:
    def __init__(self, pool, name: str = "mixed", seed: int = 0, weights=None):
        self.pool = list(pool)
        if not self.pool:
            raise ValueError("MixedOpponentPolicy needs a non-empty pool")
        self.weights = list(weights) if weights is not None else None
        if self.weights is not None:
            if len(self.weights) != len(self.pool):
                raise ValueError("MixedOpponentPolicy weights must match the pool length")
            if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError(
                    "MixedOpponentPolicy weights must be non-negative with positive sum"
                )
        self.name = name
        self._seed = seed
        self._r = random.Random(seed)
        self._prev_round = None
        self._active = self.pool[0]

    def _resample(self) -> None:
        self._active = self._r.choices(self.pool, weights=self.weights, k=1)[0]

    def reset(self, seed=None) -> None:
        s = self._seed if seed is None else seed
        self._r = random.Random(s)
        self._prev_round = None
        for p in self.pool:
            p.reset(s)

    def draft_action(self, view, legal):
        r = view.round
        if self._prev_round is None or r < self._prev_round:
            self._resample()  # new episode -> pick this episode's opponent
        self._prev_round = r
        return self._active.draft_action(view, legal)

    def battle_action(self, view, legal, state=None):
        return self._active.battle_action(view, legal, state)
