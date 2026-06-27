"""Pure self-play recording helpers for the AlphaZero pipeline.

These three functions convert PUCT search output into training targets
and steer the game during self-play generation.  They are deliberately
ml-free (numpy only) so they can be exercised in the test suite without
any torch/sb3 dependency.

A later task (P3) adds the ``record_selfplay`` game-driving generator to
this same file; the helpers below feed that driver.
"""

from __future__ import annotations

import random

import numpy as np

from locma.envs.encode import ACTION_SIZE, sem_index


def build_policy_target(view, legal: list, total) -> tuple[np.ndarray, bool]:
    """Build the AlphaZero policy target from per-edge visit counts.

    Parameters
    ----------
    view:
        A ``BattleView`` for the current position.
    legal:
        The list of legal actions in ``battle_legal(state)`` order.
    total:
        A sequence of non-negative visit counts, one per entry of ``legal``
        (same order).

    Returns
    -------
    (pi, ok)
        ``pi`` is a float32 array of shape ``(ACTION_SIZE,)``.
        ``ok`` is ``False`` when the caller should drop this row (all
        legal edges mapped to ``None``, or all mapped edges had zero
        visits); ``True`` when ``pi`` is a valid normalised distribution.
    """
    pi = np.zeros(ACTION_SIZE, dtype=np.float32)
    for i, action in enumerate(legal):
        j = sem_index(view, action)
        if j is not None and j < ACTION_SIZE:
            pi[j] += total[i]
    s = float(pi.sum())
    if s <= 0.0:
        return (pi, False)
    pi /= s
    return (pi, True)


def outcome_for(winner, seat: int) -> float:
    """Value-head training target from the moving seat's perspective.

    Returns ``+1.0`` if *winner* == *seat*, ``-1.0`` if *winner* is the
    opponent, and ``0.0`` for a draw or when *winner* is ``None``.
    """
    if winner == seat:
        return 1.0
    if winner == 1 - seat:
        return -1.0
    return 0.0


def select_move_index(total, ply: int, temp_moves: int, rng: random.Random) -> int:
    """Pick the legal index to play during self-play generation.

    Before *temp_moves* plies, samples proportionally to visit counts
    (temperature τ=1).  Afterwards returns the argmax (ties broken by
    lowest index, matching Python ``max``).

    Parameters
    ----------
    total:
        Sequence of non-negative visit counts (one per legal action).
    ply:
        Current half-move number (0-indexed).
    temp_moves:
        Number of plies over which to sample stochastically.
    rng:
        A ``random.Random`` instance — makes sampling reproducible.
    """
    if ply < temp_moves and sum(total) > 0:
        return rng.choices(range(len(total)), weights=list(total), k=1)[0]
    return int(max(range(len(total)), key=lambda i: total[i]))
