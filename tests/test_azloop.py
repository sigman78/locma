"""Tests for azloop.py: avg_hard3, h2h_winrate, and az_selfplay orchestrator.

All 8 tests are [ml]-free: they exercise the composite gate and eval helpers
via monkeypatching module-level names in azloop.
"""

from __future__ import annotations

import locma.envs.azloop as azloop

# ---------------------------------------------------------------------------
# Minimal MatchResult stub
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal MatchResult stub with just win_rate_a."""

    def __init__(self, win_rate_a: float) -> None:
        self.win_rate_a = win_rate_a


# ---------------------------------------------------------------------------
# Test 1: avg_hard3 averages win_rate_a over the three baselines
# ---------------------------------------------------------------------------


def test_avg_hard3_averages_three_baselines(monkeypatch):
    """avg_hard3 returns the mean of win_rate_a over scripted/max-guard/max-attack."""
    rates = iter([0.6, 0.7, 0.8])

    def fake_run_match(*args, **kwargs):
        return _FakeResult(next(rates))

    monkeypatch.setattr(azloop, "run_match", fake_run_match)
    monkeypatch.setattr(azloop, "make_policy", lambda spec: object())

    result = azloop.avg_hard3("model.zip", games_per_opp=5, K=2, I=5, c_puct=1.0, seed=0)
    assert abs(result - 0.7) < 1e-9, f"expected 0.7, got {result}"


# ---------------------------------------------------------------------------
# Test 2: h2h_winrate returns win_rate_a — the new net's win-rate
# ---------------------------------------------------------------------------


def test_h2h_winrate_returns_new_net_rate(monkeypatch):
    """h2h_winrate returns win_rate_a, which is the new net's fraction of wins."""
    monkeypatch.setattr(azloop, "run_match", lambda *a, **kw: _FakeResult(0.62))
    monkeypatch.setattr(azloop, "make_policy", lambda spec: object())

    result = azloop.h2h_winrate("new.zip", "best.zip", games=10, K=2, I=5, c_puct=1.0, seed=0)
    assert abs(result - 0.62) < 1e-9, f"expected 0.62, got {result}"


# ---------------------------------------------------------------------------
# Test 3: adopt when both gate conditions hold
# ---------------------------------------------------------------------------


def test_az_selfplay_adopts_when_both_conditions_hold(tmp_path, monkeypatch):
    """Net is adopted every iteration when h2h > thresh AND score >= best_score - eps."""
    warm = str(tmp_path / "warm.zip")
    prefix = str(tmp_path / "az")

    # Call 1 = initial best_score; calls 2-5 = iteration evals; call 6 = final confirm.
    avg_calls = [0]

    def fake_avg(*args, **kwargs):
        avg_calls[0] += 1
        return 0.5 if avg_calls[0] == 1 else 0.65

    monkeypatch.setattr(azloop, "avg_hard3", fake_avg)
    monkeypatch.setattr(azloop, "h2h_winrate", lambda *a, **kw: 0.6)
    monkeypatch.setattr(azloop, "record_selfplay", lambda *a, **kw: None)
    monkeypatch.setattr(azloop, "az_train", lambda *a, **kw: None)

    result = azloop.az_selfplay(
        warm_start=warm,
        prefix=prefix,
        iterations=4,
        h2h_thresh=0.53,
        hard3_eps=0.02,
        max_rejects=2,
        verbose=0,
    )

    # All 4 iterations adopted: best_net is the last iter's net.
    assert result["best_net"] == f"{prefix}-net-3.zip"
    # best_score rose above the initial 0.5.
    assert result["best_score"] > 0.5


# ---------------------------------------------------------------------------
# Test 4: reject on low h2h
# ---------------------------------------------------------------------------


def test_az_selfplay_rejects_on_low_h2h(tmp_path, monkeypatch):
    """Net is rejected when h2h <= h2h_thresh even when the metric is fine."""
    warm = str(tmp_path / "warm.zip")
    prefix = str(tmp_path / "az")

    monkeypatch.setattr(azloop, "avg_hard3", lambda *a, **kw: 0.6)
    monkeypatch.setattr(azloop, "h2h_winrate", lambda *a, **kw: 0.50)  # <= 0.53 threshold
    monkeypatch.setattr(azloop, "record_selfplay", lambda *a, **kw: None)
    monkeypatch.setattr(azloop, "az_train", lambda *a, **kw: None)

    result = azloop.az_selfplay(
        warm_start=warm,
        prefix=prefix,
        iterations=1,
        h2h_thresh=0.53,
        hard3_eps=0.02,
        max_rejects=2,
        verbose=0,
    )

    # Net rejected; best_net stays at warm_start.
    assert result["best_net"] == warm


# ---------------------------------------------------------------------------
# Test 5: reject on metric regression
# ---------------------------------------------------------------------------


def test_az_selfplay_rejects_on_metric_regression(tmp_path, monkeypatch):
    """Net rejected when score drops > hard3_eps below best_score, even with h2h pass."""
    warm = str(tmp_path / "warm.zip")
    prefix = str(tmp_path / "az")

    # initial best_score = 0.6; iteration score = 0.55 → drop of 0.05 > eps=0.02.
    avg_calls = [0]

    def fake_avg(*args, **kwargs):
        avg_calls[0] += 1
        return 0.6 if avg_calls[0] == 1 else 0.55

    monkeypatch.setattr(azloop, "avg_hard3", fake_avg)
    monkeypatch.setattr(azloop, "h2h_winrate", lambda *a, **kw: 0.6)  # h2h passes
    monkeypatch.setattr(azloop, "record_selfplay", lambda *a, **kw: None)
    monkeypatch.setattr(azloop, "az_train", lambda *a, **kw: None)

    result = azloop.az_selfplay(
        warm_start=warm,
        prefix=prefix,
        iterations=1,
        h2h_thresh=0.53,
        hard3_eps=0.02,
        max_rejects=2,
        verbose=0,
    )

    # Rejected due to regression; best_net stays at warm_start.
    assert result["best_net"] == warm


# ---------------------------------------------------------------------------
# Test 6: best_score high-water mark preserved across adoptions
# ---------------------------------------------------------------------------


def test_az_selfplay_best_score_highwater(tmp_path, monkeypatch):
    """best_score stays at the high-water when a lower-scoring net is adopted."""
    warm = str(tmp_path / "warm.zip")
    prefix = str(tmp_path / "az")

    # initial=0.5, iter0=0.7, iter1=0.68, final_confirm=0.68 (fallback)
    avg_vals = [0.5, 0.7, 0.68]
    avg_calls = [0]

    def fake_avg(*args, **kwargs):
        n = avg_calls[0]
        avg_calls[0] += 1
        return avg_vals[n] if n < len(avg_vals) else 0.68

    monkeypatch.setattr(azloop, "avg_hard3", fake_avg)
    monkeypatch.setattr(azloop, "h2h_winrate", lambda *a, **kw: 0.6)
    monkeypatch.setattr(azloop, "record_selfplay", lambda *a, **kw: None)
    monkeypatch.setattr(azloop, "az_train", lambda *a, **kw: None)

    result = azloop.az_selfplay(
        warm_start=warm,
        prefix=prefix,
        iterations=2,
        h2h_thresh=0.53,
        hard3_eps=0.02,
        max_rejects=2,
        verbose=0,
    )

    # iter1 score (0.68) >= best_score(0.7) - eps(0.02) = 0.68 → adopted; best_net advances.
    assert result["best_net"] == f"{prefix}-net-1.zip"
    # best_score stays at the high-water (0.7), not the lower iter1 score (0.68).
    assert abs(result["best_score"] - 0.7) < 1e-9, f"expected 0.7, got {result['best_score']}"


# ---------------------------------------------------------------------------
# Test 7: early stop after max_rejects consecutive rejections
# ---------------------------------------------------------------------------


def test_az_selfplay_early_stop(tmp_path, monkeypatch):
    """Loop stops early after max_rejects consecutive rejections."""
    warm = str(tmp_path / "warm.zip")
    prefix = str(tmp_path / "az")

    train_calls = []

    def fake_az_train(*args, **kwargs):
        train_calls.append(1)

    monkeypatch.setattr(azloop, "avg_hard3", lambda *a, **kw: 0.3)  # constant low score
    monkeypatch.setattr(azloop, "h2h_winrate", lambda *a, **kw: 0.50)  # always rejected
    monkeypatch.setattr(azloop, "record_selfplay", lambda *a, **kw: None)
    monkeypatch.setattr(azloop, "az_train", fake_az_train)

    result = azloop.az_selfplay(
        warm_start=warm,
        prefix=prefix,
        iterations=5,
        h2h_thresh=0.53,
        hard3_eps=0.02,
        max_rejects=2,
        verbose=0,
    )

    # Stopped after exactly 2 rejections — not all 5 iterations.
    assert len(result["history"]) == 2
    assert len(train_calls) == 2


# ---------------------------------------------------------------------------
# Test 8: next iteration uses the retained best_net after a rejection
# ---------------------------------------------------------------------------


def test_az_selfplay_next_iter_uses_retained_best_on_reject(tmp_path, monkeypatch):
    """After a rejection, record_selfplay and az_train are called with the retained best_net."""
    warm = str(tmp_path / "warm.zip")
    prefix = str(tmp_path / "az")

    # iter0: h2h=0.6 → adopt; iter1: h2h=0.50 → reject; iter2: h2h=0.6 → adopt;
    # final h2h call = 0.6 (need 4 values total)
    h2h_vals = iter([0.6, 0.50, 0.6, 0.6])
    recorded_oracle = []
    recorded_warm_start = []

    def fake_record_selfplay(oracle_path, **kwargs):
        recorded_oracle.append(oracle_path)

    def fake_az_train(datasets, warm_start, **kwargs):
        recorded_warm_start.append(warm_start)

    monkeypatch.setattr(azloop, "avg_hard3", lambda *a, **kw: 0.5)  # constant; gate driven by h2h
    monkeypatch.setattr(azloop, "h2h_winrate", lambda *a, **kw: next(h2h_vals))
    monkeypatch.setattr(azloop, "record_selfplay", fake_record_selfplay)
    monkeypatch.setattr(azloop, "az_train", fake_az_train)

    azloop.az_selfplay(
        warm_start=warm,
        prefix=prefix,
        iterations=3,
        h2h_thresh=0.53,
        hard3_eps=0.02,
        max_rejects=4,  # no early stop; run all 3 iterations
        verbose=0,
    )

    net_0 = f"{prefix}-net-0.zip"

    # iter0 calls use warm_start (initial best).
    assert recorded_oracle[0] == warm
    assert recorded_warm_start[0] == warm

    # iter1 (the rejected one) uses the adopted net-0 as oracle and warm_start.
    assert recorded_oracle[1] == net_0, (
        f"iter1 oracle: expected {net_0!r}, got {recorded_oracle[1]!r}"
    )
    assert recorded_warm_start[1] == net_0, (
        f"iter1 warm_start: expected {net_0!r}, got {recorded_warm_start[1]!r}"
    )

    # iter2 still uses net-0 (rejected net-1 was discarded).
    assert recorded_oracle[2] == net_0, (
        f"iter2 oracle: expected {net_0!r}, got {recorded_oracle[2]!r}"
    )
    assert recorded_warm_start[2] == net_0, (
        f"iter2 warm_start: expected {net_0!r}, got {recorded_warm_start[2]!r}"
    )
