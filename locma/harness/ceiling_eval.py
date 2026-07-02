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


HARD3 = ("scripted", "max-guard", "max-attack")


def _ppo_policy(model_path: str, draft_noise: int = 0):
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.drafts import (  # noqa: PLC0415
        BalancedDraftPolicy,
        PartialRandomDraftPolicy,
    )
    from locma.policies.ppo import MaskablePPOBattlePolicy  # noqa: PLC0415

    draft = BalancedDraftPolicy()
    if draft_noise:
        draft = PartialRandomDraftPolicy(draft, draft_noise)
    return Composer(MaskablePPOBattlePolicy(model_path), draft, name="ppo")


# Per-process policy cache: a pool worker evaluates many (seed) tasks for the
# same model — load the net once per (path, noise) instead of once per task.
_POLICY_CACHE: dict = {}


def _cached_ppo_policy(model_path: str, draft_noise: int = 0):
    key = (model_path, draft_noise)
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _ppo_policy(model_path, draft_noise)
    return _POLICY_CACHE[key]


def avg_hard3_at_seed(
    model_path,
    seed: int,
    games_per_seed: int,
    opponents=HARD3,
    draft_noise: int = 0,
) -> float:
    """avg-hard3 (mean win-rate over the opponents) at ONE eval seed.

    Top-level and spec-string-parameterised so it is the picklable unit of work
    for a process pool. ``draft_noise`` (k) replaces k of the PPO side's 30 draft
    picks with uniform random ones (deck-robustness probes).
    """
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    me = _cached_ppo_policy(model_path, draft_noise)
    rates = [
        run_match(me, make_policy(o), games=games_per_seed, seed=seed).win_rate_a for o in opponents
    ]
    return sum(rates) / len(rates)


def avg_hard3_per_seed(
    model_path,
    seeds,
    games_per_seed,
    opponents=HARD3,
    draft_noise: int = 0,
):
    """avg-hard3 (mean win-rate over the 3 opponents) at each eval seed."""
    return [avg_hard3_at_seed(model_path, s, games_per_seed, opponents, draft_noise) for s in seeds]


def _rate_table(paths, seeds, games_per_seed, workers: int = 1) -> dict:
    """{(path, seed): avg-hard3} for every path x seed, serial or process-parallel."""
    keys = [(p, s) for p in paths for s in seeds]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        from locma.harness.parallel import init_eval_worker  # noqa: PLC0415

        with ProcessPoolExecutor(
            max_workers=min(workers, len(keys)), initializer=init_eval_worker
        ) as ex:
            rates = list(
                ex.map(
                    avg_hard3_at_seed,
                    [p for p, _ in keys],
                    [s for _, s in keys],
                    [games_per_seed] * len(keys),
                )
            )
        return dict(zip(keys, rates, strict=True))
    return {(p, s): avg_hard3_at_seed(p, s, games_per_seed) for p, s in keys}


def run_verdict(
    candidate_paths, b0_paths, seeds, games_per_seed, threshold: float = 0.03, workers: int = 1
):
    """Paired per-seed avg-hard3 difference (candidate models minus B0 models), averaged
    over each arm's model list, with a bootstrap CI and the symmetric verdict.

    ``workers > 1`` fans the (model, seed) grid out over a process pool — results
    are identical to the serial run (each cell is an independent seeded match)."""
    table = _rate_table(list(candidate_paths) + list(b0_paths), seeds, games_per_seed, workers)

    def arm_matrix(paths):
        # rows = models, cols = seeds
        return np.array([[table[(p, s)] for s in seeds] for p in paths])

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
