"""Confidence intervals and binomial hypothesis tests."""

from __future__ import annotations

import math

from scipy.stats import binomtest


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute the Wilson confidence interval for a binomial proportion.

    Args:
        wins: Number of successes.
        n: Total number of trials.
        z: Z-score (default 1.96 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound), clamped to [0.0, 1.0].
    """
    if n == 0:
        return (0.0, 1.0)

    phat = wins / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom

    return (max(0.0, center - half), min(1.0, center + half))


def binomial_test(wins: int, n: int, p0: float = 0.5) -> float:
    """Compute the two-sided binomial test p-value.

    Args:
        wins: Number of successes.
        n: Total number of trials.
        p0: Null hypothesis probability.

    Returns:
        Two-sided p-value from scipy.stats.binomtest.
    """
    return binomtest(wins, n, p0, alternative="two-sided").pvalue
