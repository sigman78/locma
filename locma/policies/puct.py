"""Shared PUCT core with injectable oracle.

Provides the PUCT tree-search primitives used by ``azlite`` (heuristic oracle)
and, in later phases, ``netdmcts`` (net oracle).  Intentionally import-light:
no torch/sb3 — the oracle is injected so this module stays pure-Python.

Oracle protocol (duck-typed)::

    oracle.priors(sim, actions, seat) -> list[float]   # prior over actions
    oracle.value(sim, root_seat)      -> float          # leaf value in [-1, 1]

Usage::

    visit_counts = puct_search(state, oracle, iterations=100, c_puct=1.5, rng=rng)
    best_action_index = max(range(len(visit_counts)), key=lambda i: visit_counts[i])
"""

from __future__ import annotations

import math

from locma.core import battle as battlemod
from locma.core.state import Phase
from locma.policies.mcts import _clone_battle


class _Node:
    __slots__ = ("seat", "actions", "P", "N", "W", "children")

    def __init__(self, seat: int, actions: list, P: list[float]):
        self.seat = seat  # seat to move at this node
        self.actions = actions
        self.P = P  # prior per edge
        self.N = [0] * len(actions)  # edge visit count
        self.W = [0.0] * len(actions)  # edge cumulative value (ROOT-seat perspective)
        self.children = [None] * len(actions)


def _select(node: _Node, root_seat: int, c_puct: float) -> int:
    """PUCT selection: argmax Q + U.

    Q is from the ROOT seat's perspective (sign flipped for opponent nodes).
    U = c_puct * P * sqrt(ΣN) / (1 + N).
    """
    sqrt_sum = math.sqrt(sum(node.N) + 1)
    sign = 1.0 if node.seat == root_seat else -1.0  # opponent minimises root value
    best, best_score = 0, -math.inf
    for i in range(len(node.actions)):
        q = sign * (node.W[i] / node.N[i]) if node.N[i] > 0 else 0.0
        u = c_puct * node.P[i] * sqrt_sum / (1 + node.N[i])
        score = q + u
        if score > best_score:
            best, best_score = i, score
    return best


def _reward(sim, root_seat: int) -> float:
    """Terminal/turn-cap reward from ``root_seat``'s perspective: ±1 or health-diff sign."""
    if sim.winner is not None:
        return 1.0 if sim.winner == root_seat else -1.0
    me = sim.players[root_seat].health
    op = sim.players[1 - root_seat].health
    return 1.0 if me > op else (-1.0 if me < op else 0.0)


def puct_search(root_state, oracle, iterations: int, c_puct: float, rng) -> list[int]:
    """Run PUCT/AlphaZero-style search and return root edge visit counts.

    Parameters
    ----------
    root_state:
        A ``GameState`` in the BATTLE phase; the search forward-simulates from here.
        Must not be mutated (the function clones it for each simulation).
    oracle:
        An object exposing ``priors(sim, actions, seat) -> list[float]`` (prior
        distribution over ``actions`` for the player ``seat`` to move) and
        ``value(sim, root_seat) -> float`` (leaf evaluation in [-1, 1] from
        ``root_seat``'s perspective).
    iterations:
        Number of simulations to run (each backprops exactly one visit to the root).
    c_puct:
        Exploration constant in the PUCT upper-confidence bound.
    rng:
        A ``random.Random`` instance (reserved for oracle/future use; the PUCT
        core itself is deterministic given the oracle).

    Returns
    -------
    list[int]
        Root edge visit counts in the same order as ``battle_legal(root_state)``.
        ``sum(result) == iterations`` and every entry is ≥ 0.
    """
    root_seat = root_state.current
    legal = list(battlemod.battle_legal(root_state))
    root = _Node(root_seat, legal, oracle.priors(root_state, legal, root_seat))

    for _ in range(iterations):
        sim = _clone_battle(root_state)
        node = root
        path: list[tuple[_Node, int]] = []

        while True:
            ai = _select(node, root_seat, c_puct)
            path.append((node, ai))
            battlemod.apply_battle(sim, node.actions[ai])
            if sim.phase != Phase.BATTLE:
                value = _reward(sim, root_seat)
                break
            child = node.children[ai]
            if child is None:  # expand + evaluate this leaf
                actions = list(battlemod.battle_legal(sim))
                node.children[ai] = _Node(
                    sim.current, actions, oracle.priors(sim, actions, sim.current)
                )
                value = oracle.value(sim, root_seat)
                break
            node = child

        for n, ai in path:
            n.N[ai] += 1
            n.W[ai] += value

    return list(root.N)
