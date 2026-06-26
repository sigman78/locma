from __future__ import annotations

import math


def elo_update(ra: float, rb: float, score_a: float, k: float = 32) -> tuple[float, float]:
    """Update Elo ratings for two players after a single game.

    Args:
        ra: Current rating of player A.
        rb: Current rating of player B.
        score_a: Outcome for A: 1.0 (win), 0.5 (draw), 0.0 (loss).
        k: K-factor controlling rating volatility.

    Returns:
        Tuple of updated ratings (ra2, rb2).
    """
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    ra2 = ra + k * (score_a - ea)
    rb2 = rb + k * ((1 - score_a) - eb)
    return ra2, rb2


def elo_from_results(
    pairs: list[tuple[str, str, float]],
    start: float = 1500,
    reg: float = 1.0,
    max_iter: int = 10000,
    tol: float = 1e-10,
) -> dict[str, float]:
    """Compute Elo ratings via an order-free Bradley-Terry fit.

    Unlike a sequential pass of :func:`elo_update`, this estimator aggregates the
    results into a win/games summary and fits Bradley-Terry strengths by
    minorization-maximization (Hunter 2004). The result depends only on the
    *set* of games, not their order, and converges to a fixed point. The fitted
    strengths are then mapped onto an Elo scale.

    A Beta(reg, reg) prior pulls each strength toward a virtual anchor of
    strength 1, which keeps an undefeated player's strength finite (it would
    otherwise diverge).

    Args:
        pairs: List of (a_name, b_name, score_a) tuples where score_a in {0, 0.5, 1}.
        start: Mean Elo rating (the geometric-mean strength maps to this value).
        reg: Strength of the Beta(reg, reg) regularizing prior.
        max_iter: Maximum number of MM sweeps.
        tol: Stop when the max relative change in a sweep falls below this.

    Returns:
        Dict mapping every player name in ``pairs`` to its Elo rating.
    """
    names: list[str] = []
    seen: set[str] = set()
    wins: dict[str, float] = {}
    games: dict[tuple[str, str], float] = {}

    def _register(name: str) -> None:
        if name not in seen:
            seen.add(name)
            names.append(name)
            wins[name] = 0.0

    for a, b, score_a in pairs:
        _register(a)
        _register(b)
        wins[a] += score_a
        wins[b] += 1.0 - score_a
        key = (a, b) if a <= b else (b, a)
        games[key] = games.get(key, 0.0) + 1.0

    if not names:
        return {}

    # Opponents and game counts per player, for the MM update denominator.
    opponents: dict[str, list[tuple[str, float]]] = {name: [] for name in names}
    for (i, j), g in games.items():
        opponents[i].append((j, g))
        opponents[j].append((i, g))

    p: dict[str, float] = {name: 1.0 for name in names}

    for _ in range(max_iter):
        p_new: dict[str, float] = {}
        for i in names:
            num = wins[i] + reg
            den = (2.0 * reg) / (p[i] + 1.0)
            for j, g in opponents[i]:
                den += g / (p[i] + p[j])
            p_new[i] = num / den

        # Normalize so the geometric mean of strengths is 1.0.
        log_mean = sum(math.log(v) for v in p_new.values()) / len(p_new)
        norm = math.exp(log_mean)
        for i in names:
            p_new[i] /= norm

        max_rel = max(abs(p_new[i] - p[i]) / p[i] for i in names)
        p = p_new
        if max_rel < tol:
            break

    scale = 400.0 / math.log(10)
    return {name: start + scale * math.log(p[name]) for name in names}
