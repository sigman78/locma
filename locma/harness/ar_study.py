"""avg-hard3 evaluation + symmetric paired-bootstrap verdict for the
autoregressive-head study. See docs/ppo-autoreg-action-design.md."""

from __future__ import annotations

import numpy as np

HARD3: tuple[str, ...] = ("scripted", "max-guard", "max-attack")


def avg_hard3_spec(policy_spec: str, seeds, games_per_seed: int = 2) -> np.ndarray:
    """avg-hard3 for any policy spec, one value per eval seed.

    Each seed plays ``games_per_seed`` mirrored matches against each of the
    three hard opponents; the per-seed value is the mean win-rate over the
    three. Reusing the same seeds gives paired comparisons across models.
    """
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    pol = make_policy(policy_spec)
    out = np.zeros(len(seeds), dtype=np.float64)
    for i, s in enumerate(seeds):
        rates = []
        for opp in HARD3:
            res = run_match(pol, make_policy(opp), games=games_per_seed, seed=int(s))
            rates.append(res.win_rate_a)
        out[i] = float(np.mean(rates))
    return out


def hard3_per_seed(model_path: str, seeds, games_per_seed: int = 2) -> np.ndarray:
    """avg-hard3 per seed for a saved PPO model composed as ``ppo:<path>``."""
    return avg_hard3_spec(f"ppo:{model_path}", seeds, games_per_seed)


def paired_bootstrap_ci(
    diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean paired difference. Returns (lo, hi, point)."""
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = diff[rng.integers(0, n, size=n)].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi, float(diff.mean())


def decide(lo: float, hi: float, point: float, thresh: float = 0.03) -> str:
    """Symmetric verdict against +/- thresh."""
    if point >= thresh and lo > 0:
        return "ar-helps"
    if -thresh <= lo and hi <= thresh:
        return "no-help"
    return "inconclusive"


def run_verdict(flat_path: str, ar_path: str, seeds, games_per_seed: int = 2) -> dict:
    """Full paired verdict: per-seed avg-hard3 for both models, then bootstrap."""
    flat = hard3_per_seed(flat_path, seeds, games_per_seed)
    ar = hard3_per_seed(ar_path, seeds, games_per_seed)
    diff = ar - flat
    lo, hi, point = paired_bootstrap_ci(diff)
    return {
        "flat_mean": float(flat.mean()),
        "ar_mean": float(ar.mean()),
        "delta": point,
        "ci": (lo, hi),
        "verdict": decide(lo, hi, point),
        "n_seeds": len(seeds),
    }
