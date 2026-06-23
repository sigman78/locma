"""Sequential Probability Ratio Test (SPRT) using Wald's log-likelihood-ratio."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SprtResult:
    """Result of a Sequential Probability Ratio Test.

    Attributes:
        decision: One of "accept_h1", "accept_h0", or "continue".
        llr: Log-likelihood ratio.
        n: Number of observations.
    """

    decision: str
    llr: float
    n: int


def sprt(
    wins: int,
    n: int,
    p0: float,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> SprtResult:
    """Compute the Sequential Probability Ratio Test decision.

    Args:
        wins: Number of successes.
        n: Total number of observations.
        p0: Null hypothesis probability.
        p1: Alternative hypothesis probability.
        alpha: Type I error rate (default 0.05).
        beta: Type II error rate (default 0.05).

    Returns:
        SprtResult with decision, log-likelihood ratio, and sample size.
    """
    losses = n - wins
    llr = (wins * math.log(p1 / p0)) + (losses * math.log((1 - p1) / (1 - p0)))
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))

    if llr >= upper:
        return SprtResult("accept_h1", llr, n)
    if llr <= lower:
        return SprtResult("accept_h0", llr, n)
    return SprtResult("continue", llr, n)
