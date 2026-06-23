from __future__ import annotations


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


def elo_from_results(pairs: list[tuple[str, str, float]], start: float = 1500) -> dict[str, float]:
    """Compute Elo ratings from a sequence of game results.

    Args:
        pairs: List of (a_name, b_name, score_a) tuples where score_a in {0, 0.5, 1}.
        start: Starting rating for any new player.

    Returns:
        Dict mapping player name to final Elo rating.
    """
    ratings: dict[str, float] = {}
    for a, b, score_a in pairs:
        ratings.setdefault(a, start)
        ratings.setdefault(b, start)
        ratings[a], ratings[b] = elo_update(ratings[a], ratings[b], score_a)
    return ratings
