"""Network utilization diagnostics: pure-numpy metrics over activation matrices.

The instrument for the architecture-sweep prestudy: given a matrix of hidden
activations (N samples x D units) collected from a policy net on a frozen
probe dataset, quantify how much of the layer's capacity is actually used
(spectrum stats), whether units are healthy (saturation / death), how similar
two representations are (linear CKA), and how much task information is
linearly decodable (ridge probes).

Everything here is numpy-only and import-safe without the [ml] extra —
activation COLLECTION (which needs torch) lives in locma.stats.activations.

Interpretation notes
--------------------
- ``participation_ratio / width`` is the headline "fraction of the layer in
  use". Compare against the same architecture at random init (training
  typically COLLAPSES dimensionality toward task-relevant subspaces; no
  change from init means training barely shaped the layer).
- Linear probes are only meaningful relative to the same probe fit on the RAW
  observation: health/board totals are input features, so a hidden layer
  "knowing" them is not evidence of computation. Report the delta.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Spectrum statistics
# ---------------------------------------------------------------------------


def spectrum_stats(acts: np.ndarray) -> dict:
    """Eigenspectrum summary of the (centered) activation covariance.

    Parameters
    ----------
    acts: (N, D) activation matrix, N samples x D units.

    Returns a dict with:
      - ``width``: D
      - ``participation_ratio``: (sum lam)^2 / sum lam^2 over covariance
        eigenvalues lam — the classic dimensionality estimate (1..D).
      - ``pr_frac``: participation_ratio / D.
      - ``effective_rank``: Roy & Vetterli entropy rank exp(H(s/sum s)) over
        singular values s (1..D).
      - ``stable_rank``: sum(s^2) / max(s^2).
      - ``n90`` / ``n99``: smallest number of principal components explaining
        90% / 99% of the variance.
      - ``total_var``: trace of the covariance (sum of eigenvalues).
    """
    x = np.asarray(acts, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"acts must be 2-D (N, D), got shape {x.shape}")
    n, d = x.shape
    x = x - x.mean(axis=0, keepdims=True)

    # Eigenvalues of the covariance via the Gram trick on the smaller side.
    denom = max(n - 1, 1)
    if d <= n:
        cov = x.T @ x / denom
        lam = np.linalg.eigvalsh(cov)[::-1]
    else:
        gram = x @ x.T / denom
        lam = np.linalg.eigvalsh(gram)[::-1]
    lam = np.clip(lam, 0.0, None)

    total = float(lam.sum())
    if total <= 0.0:  # all-constant activations — report a degenerate spectrum
        return {
            "width": d,
            "participation_ratio": 0.0,
            "pr_frac": 0.0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "n90": 0,
            "n99": 0,
            "total_var": 0.0,
        }

    pr = total**2 / float((lam**2).sum())
    s = np.sqrt(lam)
    p = s / s.sum()
    p = p[p > 0]
    erank = float(np.exp(-(p * np.log(p)).sum()))
    stable = float(lam.sum() / lam[0])
    cum = np.cumsum(lam) / total
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    return {
        "width": d,
        "participation_ratio": float(pr),
        "pr_frac": float(pr / d),
        "effective_rank": erank,
        "stable_rank": stable,
        "n90": n90,
        "n99": n99,
        "total_var": total,
    }


# ---------------------------------------------------------------------------
# Unit health
# ---------------------------------------------------------------------------


def unit_health(acts: np.ndarray, kind: str = "tanh") -> dict:
    """Per-unit pathology summary.

    ``kind`` selects what counts as unhealthy:
      - ``"tanh"``: saturated units (|a| > 0.99 on more than half the inputs)
        plus the overall saturation rate; dead = near-zero variance.
      - ``"relu"``: dead units (active on < 1% of inputs) plus the mean duty
        cycle (fraction of inputs on which a unit is active).
      - ``"linear"``: no activation-specific pathology; dead = near-zero
        variance only.
    """
    x = np.asarray(acts, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"acts must be 2-D (N, D), got shape {x.shape}")
    d = x.shape[1]
    out: dict = {"width": d, "kind": kind}
    out["dead_frac"] = float((x.std(axis=0) < 1e-4).mean())
    if kind == "tanh":
        sat = np.abs(x) > 0.99
        out["saturated_unit_frac"] = float((sat.mean(axis=0) > 0.5).mean())
        out["saturation_rate"] = float(sat.mean())
    elif kind == "relu":
        duty = (x > 0).mean(axis=0)
        out["dead_frac"] = float((duty < 0.01).mean())
        out["mean_duty_cycle"] = float(duty.mean())
    elif kind != "linear":
        raise ValueError(f"unknown kind {kind!r} (tanh/relu/linear)")
    return out


# ---------------------------------------------------------------------------
# Representation similarity
# ---------------------------------------------------------------------------


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two activation matrices.

    Kornblith et al. 2019, feature-space form: invariant to orthogonal
    transforms and isotropic scaling of either representation. Both inputs
    must have the same N (rows aligned sample-for-sample); widths may differ.
    Returns a value in [0, 1] (1 = same representation up to rotation/scale).
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"row mismatch: {a.shape[0]} != {b.shape[0]}")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(a.T @ b) ** 2)
    denom = float(np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b))
    return cross / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Linear probes (closed-form ridge)
# ---------------------------------------------------------------------------


def _ridge_fit(x: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge with intercept via centering. Returns (W, b)."""
    xm = x.mean(axis=0, keepdims=True)
    ym = y.mean(axis=0, keepdims=True)
    xc, yc = x - xm, y - ym
    d = x.shape[1]
    w = np.linalg.solve(xc.T @ xc + l2 * np.eye(d), xc.T @ yc)
    b = ym - xm @ w
    return w, b


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def probe_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    l2: float = 1.0,
) -> dict:
    """Ridge-regression probe; returns test R^2 (vs the train-mean predictor).

    Features are standardized with TRAIN statistics, so ``l2`` is comparable
    across layers of different scale/width.
    """
    xtr = np.asarray(x_train, dtype=np.float64)
    xte = np.asarray(x_test, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64).reshape(len(xtr), -1)
    yte = np.asarray(y_test, dtype=np.float64).reshape(len(xte), -1)
    xtr, xte = _standardize(xtr, xte)
    w, b = _ridge_fit(xtr, ytr, l2)
    pred = xte @ w + b
    ss_res = float(((yte - pred) ** 2).sum())
    # R^2 against the TRAIN-mean baseline (honest out-of-sample skill).
    base = ytr.mean(axis=0, keepdims=True)
    ss_tot = float(((yte - base) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"r2": float(r2), "n_train": len(xtr), "n_test": len(xte)}


def probe_classify(
    x_train: np.ndarray,
    labels_train: np.ndarray,
    x_test: np.ndarray,
    labels_test: np.ndarray,
    n_classes: int,
    l2: float = 1.0,
    mask_test: np.ndarray | None = None,
) -> dict:
    """One-hot ridge probe ("linear probe" in the representation literature).

    Fits ridge to one-hot targets and predicts by argmax of the class scores.
    ``mask_test`` (n_test, n_classes) restricts the test-time argmax to legal
    classes — the natural setting for the teacher-action probe, where the
    legal-action mask is part of the decision. Reports accuracy plus the
    majority-class baseline computed under the same mask. For binary targets
    also reports ROC ``auc`` (rank statistic — immune to base-rate imbalance,
    the right readout when one class dominates, e.g. teacher-won ~ 0.9).
    """
    xtr = np.asarray(x_train, dtype=np.float64)
    xte = np.asarray(x_test, dtype=np.float64)
    ltr = np.asarray(labels_train, dtype=np.int64)
    lte = np.asarray(labels_test, dtype=np.int64)
    xtr, xte = _standardize(xtr, xte)

    y = np.zeros((len(ltr), n_classes))
    y[np.arange(len(ltr)), ltr] = 1.0
    w, b = _ridge_fit(xtr, y, l2)
    scores = xte @ w + b

    counts = np.bincount(ltr, minlength=n_classes).astype(np.float64)
    if mask_test is not None:
        m = np.asarray(mask_test, dtype=bool)
        scores = np.where(m, scores, -np.inf)
        base_pred = np.where(m, counts[None, :], -1.0).argmax(axis=1)
    else:
        base_pred = np.full(len(lte), int(counts.argmax()))
    pred = scores.argmax(axis=1)
    out = {
        "accuracy": float((pred == lte).mean()),
        "majority_accuracy": float((base_pred == lte).mean()),
        "n_train": len(xtr),
        "n_test": len(xte),
    }
    if n_classes == 2:
        out["auc"] = _binary_auc(scores[:, 1] - scores[:, 0], lte)
    return out


def _binary_auc(decision: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank statistic (ties get average rank)."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(decision, kind="mergesort")
    ranks = np.empty(len(decision))
    ranks[order] = np.arange(1, len(decision) + 1)
    # Average ranks over ties so equal scores contribute 0.5.
    sorted_dec = decision[order]
    _, inv, counts = np.unique(sorted_dec, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    avg = starts + (counts + 1) / 2.0
    ranks[order] = avg[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
