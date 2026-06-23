from locma.stats.intervals import binomial_test, wilson_ci
from locma.stats.sprt import sprt


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(60, 100)
    assert 0.0 <= lo < 0.6 < hi <= 1.0


def test_binomial_clear_signal():
    assert binomial_test(90, 100, 0.5) < 0.01
    assert binomial_test(50, 100, 0.5) > 0.5


def test_sprt_accepts_h1_when_dominant():
    r = sprt(95, 100, p0=0.5, p1=0.65)
    assert r.decision == "accept_h1"


def test_sprt_continue_when_ambiguous():
    r = sprt(11, 20, p0=0.5, p1=0.65)
    assert r.decision == "continue"
