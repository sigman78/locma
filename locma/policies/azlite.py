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


def _leaf_value(sim, seat: int) -> float:
    """Board/health heuristic in [-1, 1] from ``seat``'s perspective."""
    me = sim.players[seat]
    op = sim.players[1 - seat]
    h = (me.health - op.health) / 30.0
    b = (_board_power(me) - _board_power(op)) / 20.0
    v = h + 0.5 * b
    return 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)


class _Node:
    __slots__ = ("seat", "actions", "P", "N", "W", "children")

    def __init__(self, seat: int, actions: list, P: list[float]):
        self.seat = seat  # seat to move at this node
        self.actions = actions
        self.P = P  # prior per edge
        self.N = [0] * len(actions)  # edge visit count
        self.W = [0.0] * len(actions)  # edge cumulative value (ROOT-seat perspective)
        self.children = [None] * len(actions)


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
        if sim.winner is not None:
            return 1.0 if sim.winner == root_seat else -1.0
        me = sim.players[root_seat].health
        op = sim.players[1 - root_seat].health
        return 1.0 if me > op else (-1.0 if me < op else 0.0)

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

    # -- PUCT selection ------------------------------------------------------

    def _select(self, node: _Node, root_seat: int) -> int:
        sqrt_sum = math.sqrt(sum(node.N) + 1)
        sign = 1.0 if node.seat == root_seat else -1.0  # opponent minimises root value
        best, best_score = 0, -math.inf
        for i in range(len(node.actions)):
            q = sign * (node.W[i] / node.N[i]) if node.N[i] > 0 else 0.0
            u = self.c_puct * node.P[i] * sqrt_sum / (1 + node.N[i])
            score = q + u
            if score > best_score:
                best, best_score = i, score
        return best

    # -- policy interface ----------------------------------------------------

    def battle_action(self, view, legal, state=None):
        if state is None:
            raise ValueError("AZLiteBattlePolicy requires the forward-model `state` argument")
        if len(legal) == 1:
            return legal[0]

        root_seat = state.current
        root = _Node(root_seat, list(legal), self._prior(state, list(legal), root_seat))

        for _ in range(self.iterations):
            sim = _clone_battle(state)
            node = root
            path: list[tuple[_Node, int]] = []

            while True:
                ai = self._select(node, root_seat)
                path.append((node, ai))
                battlemod.apply_battle(sim, node.actions[ai])
                if sim.phase != Phase.BATTLE:
                    value = self._reward(sim, root_seat)
                    break
                child = node.children[ai]
                if child is None:  # expand + evaluate this leaf
                    actions = list(battlemod.battle_legal(sim))
                    node.children[ai] = _Node(
                        sim.current, actions, self._prior(sim, actions, sim.current)
                    )
                    value = self._value(sim, root_seat)
                    break
                node = child

            for n, ai in path:
                n.N[ai] += 1
                n.W[ai] += value

        best = max(range(len(root.actions)), key=lambda i: root.N[i])
        return root.actions[best]
