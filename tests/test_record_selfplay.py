"""Tests for record_selfplay — the AlphaZero self-play data generator.

Requires the [ml] extra (sb3_contrib, torch). All tests are gated with
``pytest.importorskip("sb3_contrib")`` so they are skipped cleanly in CI
without the extra.
"""

from __future__ import annotations

import numpy as np
import pytest

# Long model-training/game-playing tests: opt-in via `pytest -m slow`.
pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared fixture: build a tiny token MaskablePPO model once per test session
# ---------------------------------------------------------------------------


def _make_token_model(tmp_path):
    """Build and save an untrained token MaskablePPO; return the path string."""
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")

    from locma.envs.training import _build_env, _make_model  # noqa: PLC0415

    env = _build_env("random", 0, 1, obs_mode="token")
    model = _make_model(env, obs_mode="token", seed=0, verbose=0, ent_coef=0.02)
    path = str(tmp_path / "m.zip")
    model.save(path)
    env.close()
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TINY_PARAMS = dict(self_play_games=2, baseline_games=1, K=2, I=4, temp_moves=1)

_NPZ_KEYS = {
    "obs_tokens",
    "obs_card_ids",
    "obs_token_mask",
    "obs_scalars",
    "policy_target",
    "mask",
    "value_target",
    "seat",
    "game_id",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_selfplay_shapes_and_keys(tmp_path):
    """npz has all 9 keys with correct shapes/dtypes; manifest n_examples matches row count."""
    pytest.importorskip("sb3_contrib")
    from locma.envs.encode import ACTION_SIZE, MAX_TOKENS, N_TACTICAL, TOKEN_FEATS  # noqa: PLC0415
    from locma.envs.selfplay import record_selfplay  # noqa: PLC0415

    oracle_path = _make_token_model(tmp_path)
    out = str(tmp_path / "sp.npz")
    manifest = record_selfplay(oracle_path, out=out, seed=0, **_TINY_PARAMS)

    data = np.load(out)
    assert set(data.files) == _NPZ_KEYS, f"missing/extra keys: {set(data.files) ^ _NPZ_KEYS}"

    n = manifest["n_examples"]
    assert n >= 1, "expected at least one example"
    assert len(data["policy_target"]) == n

    assert data["obs_tokens"].shape == (n, MAX_TOKENS, TOKEN_FEATS)
    assert data["obs_tokens"].dtype == np.float32

    assert data["obs_card_ids"].shape == (n, MAX_TOKENS)
    assert data["obs_card_ids"].dtype == np.float32

    assert data["obs_token_mask"].shape == (n, MAX_TOKENS)
    assert data["obs_token_mask"].dtype == np.float32

    assert data["obs_scalars"].shape == (n, N_TACTICAL)
    assert data["obs_scalars"].dtype == np.float32

    assert data["policy_target"].shape == (n, ACTION_SIZE)
    assert data["policy_target"].dtype == np.float32

    assert data["mask"].shape == (n, ACTION_SIZE)
    assert data["mask"].dtype == bool

    assert data["value_target"].shape == (n,)
    assert data["value_target"].dtype == np.float32

    assert data["seat"].shape == (n,)
    assert data["seat"].dtype == np.int8

    assert data["game_id"].shape == (n,)
    assert data["game_id"].dtype == np.int32


def test_selfplay_policy_target_is_valid_distribution(tmp_path):
    """Every policy_target row sums to ~1.0 and is zero where mask is False."""
    pytest.importorskip("sb3_contrib")
    from locma.envs.selfplay import record_selfplay  # noqa: PLC0415

    oracle_path = _make_token_model(tmp_path)
    out = str(tmp_path / "sp.npz")
    record_selfplay(oracle_path, out=out, seed=0, **_TINY_PARAMS)

    data = np.load(out)
    pt = data["policy_target"]
    mask = data["mask"]

    for i in range(len(pt)):
        assert abs(pt[i].sum() - 1.0) < 1e-5, f"row {i} sums to {pt[i].sum()}"
        # Entries outside the mask must be zero
        illegal_mass = pt[i][~mask[i]].sum()
        assert illegal_mass == 0.0, f"row {i} has mass {illegal_mass} on illegal actions"


def test_selfplay_value_target_in_valid_set(tmp_path):
    """value_target values are all in {-1.0, 0.0, 1.0}."""
    pytest.importorskip("sb3_contrib")
    from locma.envs.selfplay import record_selfplay  # noqa: PLC0415

    oracle_path = _make_token_model(tmp_path)
    out = str(tmp_path / "sp.npz")
    manifest = record_selfplay(oracle_path, out=out, seed=0, **_TINY_PARAMS)

    data = np.load(out)
    vt = data["value_target"]
    valid = {-1.0, 0.0, 1.0}
    bad = [float(v) for v in vt if float(v) not in valid]
    assert not bad, f"unexpected value_target values: {bad}"

    # run_game always resolves a winner, so outcome_for never returns 0.0 on a
    # clean game — every z=0 row would be a leaked partial-game row. With no
    # failed games there must be none.
    if manifest["failed_games"] == 0:
        zeros = [float(v) for v in vt if float(v) == 0.0]
        assert not zeros, f"clean run leaked {len(zeros)} z=0 rows"


def test_selfplay_manifest_constants(tmp_path):
    """Manifest layout-guard fields match encode.py constants; obs_mode=='token'."""
    pytest.importorskip("sb3_contrib")
    from locma.envs.encode import ACTION_SIZE, MAX_TOKENS, N_TACTICAL, TOKEN_FEATS  # noqa: PLC0415
    from locma.envs.selfplay import record_selfplay  # noqa: PLC0415

    oracle_path = _make_token_model(tmp_path)
    out = str(tmp_path / "sp.npz")
    manifest = record_selfplay(oracle_path, out=out, seed=0, **_TINY_PARAMS)

    assert manifest["obs_mode"] == "token"
    assert manifest["max_tokens"] == MAX_TOKENS
    assert manifest["token_feats"] == TOKEN_FEATS
    assert manifest["n_tactical"] == N_TACTICAL
    assert manifest["action_size"] == ACTION_SIZE


def test_selfplay_determinism(tmp_path):
    """Two calls with the same seed produce identical policy_target and value_target."""
    pytest.importorskip("sb3_contrib")
    from locma.envs.selfplay import record_selfplay  # noqa: PLC0415

    oracle_path = _make_token_model(tmp_path)
    out_a = str(tmp_path / "sp_a.npz")
    out_b = str(tmp_path / "sp_b.npz")

    record_selfplay(oracle_path, out=out_a, seed=42, **_TINY_PARAMS)
    record_selfplay(oracle_path, out=out_b, seed=42, **_TINY_PARAMS)

    a = np.load(out_a)
    b = np.load(out_b)

    assert len(a["policy_target"]) == len(b["policy_target"]), (
        "same seed produced different example counts"
    )
    np.testing.assert_array_equal(a["policy_target"], b["policy_target"])
    np.testing.assert_array_equal(a["value_target"], b["value_target"])
