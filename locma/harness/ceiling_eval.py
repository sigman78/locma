"""Verdict harness for the PPO ceiling study.

Stats layer (this top half) is pure numpy — paired bootstrap CI over per-eval-seed
avg-hard3 differences + the symmetric +0.03 decision rule. The runner (bottom half)
reuses run_match to produce the deltas.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(deltas, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """Mean and (1-alpha) percentile CI of the paired differences via bootstrap.

    ``deltas[i]`` = (candidate avg-hard3 − B0 avg-hard3) at eval seed i, using common
    random numbers so the difference has much lower variance than either rate alone.
    """
    d = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot_means = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(d.mean()), float(lo), float(hi)


def decide(mean_delta: float, ci_lo: float, ci_hi: float, threshold: float = 0.03) -> str:
    """Symmetric verdict: a lift counts only if it clears the threshold AND the CI
    excludes zero. Everything else confirms the ceiling."""
    if mean_delta >= threshold and ci_lo > 0.0:
        return "headroom"
    return "ceiling-confirmed"
