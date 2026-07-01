import numpy as np

from locma.harness.ar_study import decide, paired_bootstrap_ci


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    diff = rng.normal(0.05, 0.02, size=400)
    lo, hi, point = paired_bootstrap_ci(diff, n_boot=2000, seed=0)
    assert lo < point < hi
    assert abs(point - diff.mean()) < 1e-9


def test_decide_verdicts():
    # clear headroom: point >= +0.03 and CI excludes 0
    assert decide(0.02, 0.08, 0.05) == "ar-helps"
    # tight around zero, within +/-0.03 both sides
    assert decide(-0.01, 0.02, 0.005) == "no-help"
    # wide / straddling the band
    assert decide(-0.05, 0.06, 0.01) == "inconclusive"


def test_avg_hard3_spec_scripted_shape_and_range():
    from locma.harness.ar_study import avg_hard3_spec  # noqa: PLC0415

    out = avg_hard3_spec("scripted", seeds=[10, 11, 12], games_per_seed=1)
    assert out.shape == (3,)
    assert ((out >= 0.0) & (out <= 1.0)).all()
