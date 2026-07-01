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


def _ppo_policy(model_path: str):
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
    from locma.policies.ppo import MaskablePPOBattlePolicy  # noqa: PLC0415

    return Composer(MaskablePPOBattlePolicy(model_path), BalancedDraftPolicy(), name="ppo")


def avg_hard3_per_seed(
    model_path,
    seeds,
    games_per_seed,
    opponents=("scripted", "max-guard", "max-attack"),
):
    """avg-hard3 (mean win-rate over the 3 opponents) at each eval seed."""
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    me = _ppo_policy(model_path)
    opps = [make_policy(o) for o in opponents]
    out = []
    for s in seeds:
        rates = [run_match(me, opp, games=games_per_seed, seed=s).win_rate_a for opp in opps]
        out.append(sum(rates) / len(rates))
    return out


def run_verdict(candidate_paths, b0_paths, seeds, games_per_seed, threshold: float = 0.03):
    """Paired per-seed avg-hard3 difference (candidate models minus B0 models), averaged
    over each arm's model list, with a bootstrap CI and the symmetric verdict."""

    def arm_matrix(paths):
        # rows = models, cols = seeds
        return np.array([avg_hard3_per_seed(p, seeds, games_per_seed) for p in paths])

    cand = arm_matrix(candidate_paths).mean(axis=0)  # per-seed mean over candidate models
    b0 = arm_matrix(b0_paths).mean(axis=0)
    deltas = cand - b0
    mean_delta, lo, hi = paired_bootstrap_ci(deltas)
    return {
        "mean_delta": mean_delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "verdict": decide(mean_delta, lo, hi, threshold),
        "cand_avg": float(cand.mean()),
        "b0_avg": float(b0.mean()),
    }
