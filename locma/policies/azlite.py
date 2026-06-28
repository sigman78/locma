"""AlphaZero-lite: PUCT-guided MCTS with a heuristic (policy, value) oracle.

A drop-in battle policy that runs AlphaZero-style search on the engine's
perfect-information forward model — but instead of a self-play-trained network it
reuses EXISTING heuristics as the ``f(s) -> (policy, value)`` oracle:

- **prior** over a node's legal actions = a 1-ply heuristic lookahead: apply each
  action and score the resulting position with the same board/health leaf value
  the MCTS kit already uses, then softmax (temperature ``tau``).
- **value** at a leaf = that board/health heuristic directly (``rollout_turns=0``,
  the default — fast and deterministic), or a short heuristic rollout
  (``rollout_turns>0``) for a less myopic estimate.

PUCT selection focuses simulations on prior-promising actions, so the search
reaches MCTS-strength play at modest iteration counts while staying a clean
drop-in (`azlite:iters,c_puct,...`). Perfect-information (cheating) like
``MCTSBattlePolicy``; deterministic from the state when ``rollout_turns=0``.
"""

from __future__ import annotations

import math
import random

from locma.core import battle as battlemod
from locma.core.state import Phase
from locma.policies.battles import RandomBattlePolicy
from locma.policies.mcts import _board_power, _clone_battle
from locma.policies.puct import _reward as _puct_reward
from locma.policies.puct import puct_search


def _leaf_value(sim, seat: int) -> float:
    """Board/health heuristic in [-1, 1] from ``seat``'s perspective."""
    me = sim.players[seat]
    op = sim.players[1 - seat]
    h = (me.health - op.health) / 30.0
    b = (_board_power(me) - _board_power(op)) / 20.0
    v = h + 0.5 * b
    return 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)


class _HeuristicOracle:
    """Thin adapter exposing AZLiteBattlePolicy's _prior/_value as the oracle protocol."""

    def __init__(self, policy: AZLiteBattlePolicy):
        self._p = policy

    def priors(self, sim, actions: list, seat: int) -> list[float]:
        return self._p._prior(sim, actions, seat)

    def value(self, sim, root_seat: int) -> float:
        return self._p._value(sim, root_seat)


class AZLiteBattlePolicy:
    def __init__(
        self,
        name: str = "azlite",
        iterations: int = 100,
        c_puct: float = 1.5,
        seed: int = 0,
        rollout_turns: int = 0,
        tau: float = 0.4,
        turn_cap: int = 200,
    ):
        self.name = name
        self.iterations = iterations
        self.c_puct = c_puct
        self._seed = seed
        self.rollout_turns = rollout_turns
        self.tau = tau
        self.turn_cap = turn_cap
        self._rollout = RandomBattlePolicy("azlite-rollout", seed=seed)
        self._r = random.Random(seed)

    def reset(self, seed=None):
        s = self._seed if seed is None else seed
        self._r = random.Random(s)
        self._rollout.reset(s)

    # -- oracle: prior + value ----------------------------------------------

    def _prior(self, sim, actions, seat: int) -> list[float]:
        """1-ply heuristic lookahead softmax: how good each action looks for `seat`."""
        if len(actions) == 1:
            return [1.0]
        vals = []
        for a in actions:
            s2 = _clone_battle(sim)
            battlemod.apply_battle(s2, a)
            vals.append(_leaf_value(s2, seat))
        m = max(vals)
        exps = [math.exp((v - m) / self.tau) for v in vals]
        z = sum(exps)
        return [e / z for e in exps]

    def _reward(self, sim, root_seat: int) -> float:
        return _puct_reward(sim, root_seat)

    def _value(self, sim, root_seat: int) -> float:
        if self.rollout_turns <= 0:
            return _leaf_value(sim, root_seat)
        # short heuristic rollout: random-play a few turn boundaries, then evaluate.
        tc = 0
        while sim.phase == Phase.BATTLE and sim.turn <= self.turn_cap and tc < self.rollout_turns:
            owner = sim.current
            legal = battlemod.battle_legal(sim)
            battlemod.apply_battle(sim, self._rollout.battle_action(None, legal))
            if sim.current != owner:
                tc += 1
        if sim.phase != Phase.BATTLE:
            return self._reward(sim, root_seat)
        return _leaf_value(sim, root_seat)

    # -- policy interface ----------------------------------------------------

    def battle_action(self, view, legal, state=None):
        if state is None:
            raise ValueError("AZLiteBattlePolicy requires the forward-model `state` argument")
        if len(legal) == 1:
            return legal[0]

        oracle = _HeuristicOracle(self)
        visit_counts = puct_search(state, oracle, self.iterations, self.c_puct, self._r)
        root_actions = list(battlemod.battle_legal(state))
        best = max(range(len(root_actions)), key=lambda i: visit_counts[i])
        return root_actions[best]
