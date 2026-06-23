from __future__ import annotations

from openskill.models import PlackettLuce


def ordinal(mu: float, sigma: float) -> float:
    """Conservative skill estimate: mu - 3*sigma."""
    return mu - 3 * sigma


def openskill_from_results(pairs: list[tuple[str, str, float]]) -> dict[str, tuple[float, float]]:
    """Compute openskill (PlackettLuce) ratings from (a, b, score_a) results.

    score_a == 1.0 -> a won, 0.0 -> b won, 0.5 -> draw.
    Returns name -> (mu, sigma).
    """
    model = PlackettLuce()
    ratings: dict[str, object] = {}

    def get(name: str):
        if name not in ratings:
            ratings[name] = model.rating(name=name)
        return ratings[name]

    for a, b, score_a in pairs:
        ra, rb = get(a), get(b)
        if score_a == 0.5:
            ranks = [0, 0]
        elif score_a >= 1.0:
            ranks = [0, 1]  # a first (winner)
        else:
            ranks = [1, 0]  # b first
        [[ra2], [rb2]] = model.rate([[ra], [rb]], ranks=ranks)
        ratings[a], ratings[b] = ra2, rb2

    return {name: (float(r.mu), float(r.sigma)) for name, r in ratings.items()}
