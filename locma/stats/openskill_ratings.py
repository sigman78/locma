from __future__ import annotations

import random

from openskill.models import PlackettLuce


def ordinal(mu: float, sigma: float) -> float:
    """Conservative skill estimate: mu - 3*sigma."""
    return mu - 3 * sigma


def _single_pass(pairs: list[tuple[str, str, float]]) -> dict[str, tuple[float, float]]:
    """One online PlackettLuce sweep over ``pairs`` in the given order."""
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


def openskill_from_results(
    pairs: list[tuple[str, str, float]],
    passes: int = 32,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Compute order-free openskill (PlackettLuce) ratings from results.

    The online PlackettLuce update is order-dependent. To remove that bias we
    run it over many seeded random shufflings of ``pairs`` and average each
    player's final (mu, sigma). With a fixed ``seed`` the output is fully
    deterministic.

    score_a == 1.0 -> a won, 0.0 -> b won, 0.5 -> draw.
    Returns name -> (mean mu, mean sigma).
    """
    rng = random.Random(seed)
    sums: dict[str, tuple[float, float]] = {}
    count = 0

    for _ in range(passes):
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        result = _single_pass(shuffled)
        for name, (mu, sigma) in result.items():
            cur = sums.get(name, (0.0, 0.0))
            sums[name] = (cur[0] + mu, cur[1] + sigma)
        count += 1

    if count == 0:
        return {}

    return {name: (mu / count, sigma / count) for name, (mu, sigma) in sums.items()}
