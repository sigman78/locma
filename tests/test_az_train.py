"""Tests for az_train.py: load_selfplay loader + az_train smoke.

Tests 1–2 are pure (no ``[ml]`` extra needed).
Test 3 requires ``sb3_contrib`` / ``torch`` and is gated with
``pytest.importorskip``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from locma.envs.encode import ACTION_SIZE, MAX_TOKENS, N_TACTICAL, TOKEN_FEATS
from locma.envs.practicum import _manifest_path

_SELFPLAY_MANIFEST = {
    "obs_mode": "token",
    "max_tokens": MAX_TOKENS,
    "token_feats": TOKEN_FEATS,
    "n_tactical": N_TACTICAL,
    "action_size": ACTION_SIZE,
}

# Legal slots used in synthetic data (first N_LEGAL slots are legal).
_N_LEGAL = 8


def _write_fake_selfplay(path, n: int = 20, n_games: int = 4) -> None:
    """Write a synthetic self-play ``.npz`` + manifest for testing.

    Each ``policy_target`` row is uniform over the first ``_N_LEGAL`` actions;
    ``value_target`` cycles through {-1, 0, 1}.
    """
    rng = np.random.default_rng(42)
    rows_per_game = n // n_games
    game_id = np.repeat(np.arange(n_games, dtype=np.int32), rows_per_game)
    # Pad tail if n is not divisible by n_games.
    if len(game_id) < n:
        game_id = np.concatenate([game_id, np.full(n - len(game_id), n_games - 1, dtype=np.int32)])

    policy_target = np.zeros((n, ACTION_SIZE), dtype=np.float32)
    policy_target[:, :_N_LEGAL] = 1.0 / _N_LEGAL
    mask = np.zeros((n, ACTION_SIZE), dtype=bool)
    mask[:, :_N_LEGAL] = True
    value_target = np.tile(np.array([-1.0, 0.0, 1.0], dtype=np.float32), n // 3 + 1)[:n]

    np.savez(
        path,
        obs_tokens=rng.standard_normal((n, MAX_TOKENS, TOKEN_FEATS)).astype(np.float32),
        obs_card_ids=rng.random((n, MAX_TOKENS)).astype(np.float32),
        obs_token_mask=np.ones((n, MAX_TOKENS), dtype=np.float32),
        obs_scalars=rng.standard_normal((n, N_TACTICAL)).astype(np.float32),
        policy_target=policy_target,
        mask=mask,
        value_target=value_target,
        seat=np.zeros(n, dtype=np.int8),
        game_id=game_id,
    )
    with open(_manifest_path(str(path)), "w", encoding="utf-8") as f:
        json.dump(_SELFPLAY_MANIFEST, f)


# ---------------------------------------------------------------------------
# Test 1: load_selfplay raises ValueError on layout mismatch
# ---------------------------------------------------------------------------


def test_load_selfplay_rejects_wrong_action_size(tmp_path):
    """load_selfplay raises ValueError when action_size in manifest is wrong."""
    from locma.envs.az_train import load_selfplay  # noqa: PLC0415

    p = tmp_path / "bad.npz"
    _write_fake_selfplay(p, n=20, n_games=4)
    bad_manifest = dict(_SELFPLAY_MANIFEST, action_size=99)
    with open(_manifest_path(str(p)), "w", encoding="utf-8") as f:
        json.dump(bad_manifest, f)
    with pytest.raises(ValueError, match="layout"):
        load_selfplay(str(p))


# ---------------------------------------------------------------------------
# Test 2: load_selfplay concat produces disjoint game_ids
# ---------------------------------------------------------------------------


def test_load_selfplay_concat_disjoint_game_ids(tmp_path):
    """Two files each with game_id in {0,1} → 4 distinct ids after concat."""
    from locma.envs.az_train import load_selfplay  # noqa: PLC0415

    pa = tmp_path / "a.npz"
    pb = tmp_path / "b.npz"
    n = 8  # 8 rows / 2 games = 4 rows per game
    _write_fake_selfplay(pa, n=n, n_games=2)
    _write_fake_selfplay(pb, n=n, n_games=2)

    arrays, manifest = load_selfplay([str(pa), str(pb)])

    # Total rows = 2 * n.
    assert len(arrays["game_id"]) == 2 * n, f"expected {2 * n} rows, got {len(arrays['game_id'])}"

    # File a has game_ids {0, 1}; file b is offset to {2, 3}.
    unique_ids = set(int(x) for x in arrays["game_id"])
    assert unique_ids == {0, 1, 2, 3}, f"expected {{0,1,2,3}}, got {sorted(unique_ids)}"

    # Other array lengths should also match.
    for key in ("obs_tokens", "policy_target", "value_target"):
        assert len(arrays[key]) == 2 * n, f"key {key!r}: expected {2 * n}, got {len(arrays[key])}"


# ---------------------------------------------------------------------------
# Test 3: az_train smoke (requires sb3_contrib + torch)
# ---------------------------------------------------------------------------


def test_az_train_smoke(tmp_path):
    """az_train returns finite val metrics and train loss decreases over epochs."""
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("torch")

    import os  # noqa: PLC0415

    from locma.envs.az_train import az_train  # noqa: PLC0415
    from locma.envs.training import _build_env, _make_model  # noqa: PLC0415

    # Build and save a tiny warm-start model.
    env = _build_env("random", 0, 1, obs_mode="token")
    model = _make_model(env, obs_mode="token", seed=0, verbose=0, ent_coef=0.02)
    warm_start = str(tmp_path / "warm.zip")
    model.save(warm_start)
    env.close()

    # Build synthetic self-play data by hand (keeps the test fast and isolated).
    n = 40
    n_games = 4  # val_frac=0.5 → 2 val games, 2 train games = 20 train rows
    data_path = str(tmp_path / "sp.npz")
    _write_fake_selfplay(tmp_path / "sp.npz", n=n, n_games=n_games)

    out_path = str(tmp_path / "az.zip")
    result = az_train(
        data_path,
        warm_start,
        out=out_path,
        epochs=3,
        batch=16,
        lr=1e-3,  # higher than default to ensure a clear learning signal in 3 epochs
        val_frac=0.5,  # ensures val games are non-empty (2 of 4 games → 20 val rows)
        verbose=1,
    )

    # Output file must exist.
    assert os.path.exists(out_path), "az.zip was not saved"

    # Return dict keys.
    assert "val_policy_ce" in result
    assert "val_value_mse" in result
    assert "n_train" in result
    assert "n_val" in result
    assert "epoch_losses" in result

    # Val metrics must be finite (non-empty val set).
    assert math.isfinite(result["val_policy_ce"]), (
        f"val_policy_ce not finite: {result['val_policy_ce']}"
    )
    assert math.isfinite(result["val_value_mse"]), (
        f"val_value_mse not finite: {result['val_value_mse']}"
    )

    # Split sanity.
    assert result["n_train"] > 0
    assert result["n_val"] > 0

    # Learning must happen: last epoch mean loss < first epoch mean loss.
    losses = result["epoch_losses"]
    assert len(losses) == 3
    assert losses[-1] < losses[0], (
        f"expected train loss to decrease over 3 epochs; "
        f"epoch_1={losses[0]:.4f}, epoch_3={losses[-1]:.4f}"
    )
