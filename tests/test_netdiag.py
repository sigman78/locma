"""Tests for locma.stats.netdiag — pure numpy, no [ml] extra required."""

from __future__ import annotations

import numpy as np
import pytest

from locma.stats.netdiag import (
    linear_cka,
    probe_classify,
    probe_regression,
    spectrum_stats,
    unit_health,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# spectrum_stats
# ---------------------------------------------------------------------------


def test_spectrum_isotropic_gaussian_uses_full_width():
    x = RNG.standard_normal((5000, 32))
    s = spectrum_stats(x)
    assert s["width"] == 32
    assert s["participation_ratio"] > 28  # ~= width for isotropic data
    assert s["pr_frac"] > 0.85
    assert s["effective_rank"] > 28
    assert s["n99"] >= 30


def test_spectrum_rank_one():
    u = RNG.standard_normal((2000, 1))
    v = RNG.standard_normal((1, 16))
    s = spectrum_stats(u @ v)
    assert s["participation_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert s["stable_rank"] == pytest.approx(1.0, abs=1e-6)
    assert s["n90"] == 1
    assert s["n99"] == 1


def test_spectrum_rank_two_n99():
    x = np.zeros((1000, 8))
    x[:, 0] = RNG.standard_normal(1000) * 10
    x[:, 1] = RNG.standard_normal(1000) * 10
    s = spectrum_stats(x)
    assert s["n99"] == 2
    assert 1.5 < s["participation_ratio"] < 2.5


def test_spectrum_constant_input_degenerate():
    s = spectrum_stats(np.ones((100, 8)))
    assert s["participation_ratio"] == 0.0
    assert s["effective_rank"] == 0.0
    assert s["total_var"] == 0.0


def test_spectrum_gram_trick_matches_direct():
    # More units than samples exercises the N x N gram branch.
    x = RNG.standard_normal((50, 200))
    wide = spectrum_stats(x)
    # Same data, transposed computation path (D <= N branch) on padded copy.
    narrow = spectrum_stats(np.hstack([x, np.zeros((50, 0))]))
    assert wide["participation_ratio"] == pytest.approx(narrow["participation_ratio"])
    assert wide["participation_ratio"] < 50  # rank limited by sample count


def test_spectrum_rejects_bad_shape():
    with pytest.raises(ValueError):
        spectrum_stats(np.zeros(10))


# ---------------------------------------------------------------------------
# unit_health
# ---------------------------------------------------------------------------


def test_unit_health_tanh_saturation():
    n = 1000
    healthy = np.tanh(RNG.standard_normal((n, 3)))
    saturated = np.sign(RNG.standard_normal((n, 1)))  # |a| = 1 always
    dead = np.zeros((n, 1))
    h = unit_health(np.hstack([healthy, saturated, dead]), kind="tanh")
    assert h["saturated_unit_frac"] == pytest.approx(1 / 5)
    assert h["dead_frac"] == pytest.approx(1 / 5)
    assert 0.0 < h["saturation_rate"] < 0.5


def test_unit_health_relu_dead_units():
    n = 1000
    active = np.maximum(RNG.standard_normal((n, 3)), 0)
    dead = np.zeros((n, 2))
    h = unit_health(np.hstack([active, dead]), kind="relu")
    assert h["dead_frac"] == pytest.approx(2 / 5)
    assert 0.2 < h["mean_duty_cycle"] < 0.4  # 3 units at ~0.5 duty, 2 at 0


def test_unit_health_linear_kind_and_bad_kind():
    h = unit_health(RNG.standard_normal((100, 4)), kind="linear")
    assert h["dead_frac"] == 0.0
    with pytest.raises(ValueError):
        unit_health(np.zeros((10, 2)), kind="sigmoid")


# ---------------------------------------------------------------------------
# linear_cka
# ---------------------------------------------------------------------------


def test_cka_self_similarity_is_one():
    x = RNG.standard_normal((500, 16))
    assert linear_cka(x, x) == pytest.approx(1.0)


def test_cka_invariant_to_rotation_and_scale():
    x = RNG.standard_normal((500, 16))
    q, _ = np.linalg.qr(RNG.standard_normal((16, 16)))
    assert linear_cka(x, 3.7 * x @ q) == pytest.approx(1.0)


def test_cka_independent_data_near_zero():
    x = RNG.standard_normal((2000, 16))
    y = RNG.standard_normal((2000, 16))
    assert linear_cka(x, y) < 0.05


def test_cka_row_mismatch_raises():
    with pytest.raises(ValueError):
        linear_cka(np.zeros((10, 2)), np.zeros((11, 2)))


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def test_probe_regression_recovers_linear_target():
    x = RNG.standard_normal((2000, 10))
    w = RNG.standard_normal(10)
    y = x @ w + 0.01 * RNG.standard_normal(2000)
    r = probe_regression(x[:1500], y[:1500], x[1500:], y[1500:])
    assert r["r2"] > 0.99


def test_probe_regression_no_signal_r2_near_zero():
    x = RNG.standard_normal((2000, 10))
    y = RNG.standard_normal(2000)
    r = probe_regression(x[:1500], y[:1500], x[1500:], y[1500:])
    assert abs(r["r2"]) < 0.05


def test_probe_classify_separable():
    labels = RNG.integers(0, 4, size=2000)
    centers = RNG.standard_normal((4, 8)) * 5
    x = centers[labels] + 0.1 * RNG.standard_normal((2000, 8))
    r = probe_classify(x[:1500], labels[:1500], x[1500:], labels[1500:], n_classes=4)
    assert r["accuracy"] > 0.99
    assert r["majority_accuracy"] < 0.5


def test_probe_classify_binary_reports_auc():
    # 90/10 imbalance: accuracy hugs the base rate but AUC must see the signal.
    rng = np.random.default_rng(42)  # local rng: shared-RNG state is order-dependent
    n = 4000
    labels = (rng.random(n) < 0.1).astype(np.int64)
    x = rng.standard_normal((n, 6))
    x[:, 0] += labels * 1.5
    r = probe_classify(x[:3000], labels[:3000], x[3000:], labels[3000:], n_classes=2)
    assert "auc" in r
    assert r["auc"] > 0.8
    # No-signal AUC sits near chance.
    y2 = (rng.random(n) < 0.1).astype(np.int64)
    r2 = probe_classify(x[:3000], y2[:3000], x[3000:], y2[3000:], n_classes=2)
    assert abs(r2["auc"] - 0.5) < 0.06


def test_probe_classify_mask_restricts_argmax():
    # Features carry NO signal; a mask allowing only the true label per row
    # must force 100% accuracy, proving the argmax honors the mask.
    n = 200
    labels = RNG.integers(0, 5, size=n)
    x = RNG.standard_normal((n, 6))
    mask = np.zeros((n // 2, 5), dtype=bool)
    mask[np.arange(n // 2), labels[n // 2 :]] = True
    r = probe_classify(
        x[: n // 2], labels[: n // 2], x[n // 2 :], labels[n // 2 :], n_classes=5, mask_test=mask
    )
    assert r["accuracy"] == 1.0
    assert r["majority_accuracy"] == 1.0
