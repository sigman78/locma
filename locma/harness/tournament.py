from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from locma.harness.match import run_match
from locma.stats.intervals import binomial_test
from locma.stats.ratings import elo_from_results


@dataclass
class TournamentResult:
    policies: list[str]
    win_matrix: dict[tuple[str, str], float]
    ratings: dict[str, float]
    p_vs_reference: dict[str, float]


def run_tournament(
    policies: list,
    games: int = 50,
    seed: int = 0,
    reference: str | None = None,
    shared_draft: bool = False,
) -> TournamentResult:
    """Run a round-robin tournament among all policies.

    For each unordered pair, calls run_match(a, b, games, seed).
    Builds a win-rate matrix, Elo ratings, and (if reference given)
    a dict of binomial p-values for each non-reference policy vs the reference.

    Args:
        policies: List of policy objects with a `.name` attribute.
        games: Number of game-pairs per match (passed to run_match).
        seed: Random seed for match reproducibility.
        reference: Name of the reference policy for p_vs_reference.
        shared_draft: play every match under the shared draft variant (picks
            deplete the offer; first pick alternates by round).

    Returns:
        TournamentResult with policies, win_matrix, ratings, p_vs_reference.
    """
    names = [p.name for p in policies]
    win_matrix: dict[tuple[str, str], float] = {}
    elo_pairs: list[tuple[str, str, float]] = []
    # wins and games accumulated by each non-reference policy against the reference
    totals: dict[str, list[int]] = {n: [0, 0] for n in names}

    for a, b in combinations(policies, 2):
        res = run_match(a, b, games=games, seed=seed, shared_draft=shared_draft)
        win_matrix[(a.name, b.name)] = res.win_rate_a
        win_matrix[(b.name, a.name)] = res.wins_b / res.games

        # Expand wins into individual game records for Elo
        for _ in range(res.wins_a):
            elo_pairs.append((a.name, b.name, 1.0))
        for _ in range(res.wins_b):
            elo_pairs.append((a.name, b.name, 0.0))

        # Accumulate wins for reference comparison
        if reference in (a.name, b.name):
            other = b if a.name == reference else a
            wins_other = res.wins_b if a.name == reference else res.wins_a
            totals[other.name][0] += wins_other
            totals[other.name][1] += res.games

    ratings = elo_from_results(elo_pairs)
    for n in names:
        ratings.setdefault(n, 1500.0)

    p_vs_reference: dict[str, float] = {}
    if reference:
        for n in names:
            if n == reference:
                continue
            w, g = totals[n]
            p_vs_reference[n] = binomial_test(w, g, 0.5) if g else 1.0

    return TournamentResult(names, win_matrix, ratings, p_vs_reference)
