"""Prioritized Fictitious Self-Play (PFSP) mixture opponent (E36).

Adapts the population/fictitious-play idea from ByteRL (arXiv 2303.04096,
"Optimistic Smooth Fictitious Play"): instead of training against a fixed
scripted zoo, best-respond to a POOL of frozen opponents sampled per game,
weighted toward the ones the current policy is losing to (the paper's win-rate
weighting). Here the mixture is the training opponent; the driver
(scripts/e36_pfsp.py) grows the pool and updates the weights across generations.

The pool is a JSON file (rewritten each generation) of ``[{"spec", "weight"},
...]`` — each ``spec`` is any registry policy (a past-self ``ppo:`` checkpoint,
or a scripted opponent / ``boardkeep`` exploiter kept in the pool so self-play
never forgets the known holes). One member is sampled per game in ``reset``
(both run_game and BattleEnv.reset call it per episode), so the opponent identity
is fixed within a game and resampled between games — deterministically from the
episode seed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


class PFSPBattleMixture:
    """Battle policy that samples a frozen pool member per game, weight-prioritized.

    Goes in a Composer's battle slot; the draft half is supplied separately (the
    training env overrides it with ldraft anyway). Members are the ``.battle``
    halves of ``make_policy(spec)`` for each pool entry.
    """

    def __init__(self, pool_json: str, seed: int = 0) -> None:
        from locma.policies.registry import make_policy  # noqa: PLC0415 — avoid import cycle

        entries = json.loads(Path(pool_json).read_text())
        if not entries:
            raise ValueError(f"PFSP pool {pool_json!r} is empty")
        self._specs = [e["spec"] for e in entries]
        self._battles = [make_policy(s).battle for s in self._specs]
        w = [float(e.get("weight", 1.0)) for e in entries]
        total = sum(w) or 1.0
        self._weights = [x / total for x in w]
        self._rng = random.Random(seed)
        self._active = self._battles[0]
        self.name = f"pfsp[{len(self._battles)}]"

    def reset(self, seed: int | None = None) -> None:
        # deterministic per-episode resample from the episode seed
        rng = random.Random(seed) if seed is not None else self._rng
        i = rng.choices(range(len(self._battles)), weights=self._weights)[0]
        self._active = self._battles[i]
        if hasattr(self._active, "reset"):
            self._active.reset(seed)

    def battle_action(self, view, legal, state=None):
        return self._active.battle_action(view, legal, state)
