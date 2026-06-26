import math
import random

from locma.stats.intervals import binomial_test, wilson_ci
from locma.stats.ratings import elo_from_results
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


def _round_robin_pairs() -> list[tuple[str, str, float]]:
    # A > B > C cleanly: A beats B and C, B beats C.
    pairs: list[tuple[str, str, float]] = []
    pairs += [("A", "B", 1.0) for _ in range(5)]
    pairs += [("A", "C", 1.0) for _ in range(5)]
    pairs += [("B", "C", 1.0) for _ in range(5)]
    return pairs


def test_elo_from_results_order_independent():
    pairs = _round_robin_pairs()
    base = elo_from_results(pairs)
    shuffled = list(pairs)
    random.Random(12345).shuffle(shuffled)
    other = elo_from_results(shuffled)
    assert set(base) == set(other)
    for name in base:
        assert math.isclose(base[name], other[name], abs_tol=1e-6)


def test_elo_from_results_recovers_transitive_order():
    ratings = elo_from_results(_round_robin_pairs())
    assert ratings["A"] > ratings["B"] > ratings["C"]


def test_elo_from_results_undefeated_is_finite_and_top():
    pairs: list[tuple[str, str, float]] = []
    # A undefeated against both B and C.
    pairs += [("A", "B", 1.0) for _ in range(5)]
    pairs += [("A", "C", 1.0) for _ in range(5)]
    # B and C each beat the other sometimes.
    pairs += [("B", "C", 1.0) for _ in range(3)]
    pairs += [("C", "B", 1.0) for _ in range(3)]
    ratings = elo_from_results(pairs)
    assert math.isfinite(ratings["A"])
    assert ratings["A"] == max(ratings.values())
