import numpy as np

from locma.harness.ceiling_eval import decide, paired_bootstrap_ci


def test_bootstrap_ci_brackets_clear_positive_signal():
    deltas = np.full(40, 0.05)
    mean, lo, hi = paired_bootstrap_ci(deltas, n_boot=2000, seed=0)
    assert abs(mean - 0.05) < 1e-9 and lo > 0.0


def test_decide_headroom_when_above_threshold_and_ci_excludes_zero():
    assert decide(0.05, 0.02, 0.08) == "headroom"


def test_decide_ceiling_when_within_noise():
    assert decide(0.005, -0.02, 0.03) == "ceiling-confirmed"


def test_decide_ceiling_when_point_high_but_ci_crosses_zero():
    # Big point estimate but the CI includes 0 → not resolved → ceiling-confirmed.
    assert decide(0.04, -0.01, 0.09) == "ceiling-confirmed"
