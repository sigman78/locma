# tests/test_distill_token.py
"""Token-obs mode behavior_clone smoke tests (require the [ml] extra).

Pure load_practicum tests (no ML deps) live in test_distill_token_load.py.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from locma.envs.encode import ACTION_SIZE, MAX_TOKENS, N_TACTICAL, TOKEN_FEATS
from locma.envs.practicum import _manifest_path

# ---------------------------------------------------------------------------
# Helpers (no ML imports needed here)
# ---------------------------------------------------------------------------

_TOKEN_MANIFEST = {
    "obs_mode": "token",
    "max_tokens": MAX_TOKENS,
    "token_feats": TOKEN_FEATS,
    "n_tactical": N_TACTICAL,
    "action_size": ACTION_SIZE,
}


def _write_fake_token_practicum(path, n: int = 20, n_games: int = 5) -> None:
    """Write a minimal token-mode practicum npz + manifest for testing."""
    rng = np.random.default_rng(42)
    game_id = np.repeat(np.arange(n_games, dtype=np.int32), n // n_games)
    np.savez(
        path,
        obs_tokens=rng.standard_normal((n, MAX_TOKENS, TOKEN_FEATS)).astype(np.float32),
        obs_card_ids=rng.random((n, MAX_TOKENS)).astype(np.float32),
        obs_token_mask=np.ones((n, MAX_TOKENS), dtype=np.float32),
        obs_scalars=rng.standard_normal((n, N_TACTICAL)).astype(np.float32),
        action=rng.integers(0, ACTION_SIZE, size=n).astype(np.int64),
        mask=np.ones((n, ACTION_SIZE), dtype=bool),
        winner=np.zeros(n, dtype=np.int8),
        seat=np.zeros(n, dtype=np.int8),
        opponent_id=np.zeros(n, dtype=np.int8),
        game_id=game_id,
    )
    with open(_manifest_path(str(path)), "w", encoding="utf-8") as f:
        json.dump(_TOKEN_MANIFEST, f)


# ---------------------------------------------------------------------------
# Item 1 — manifest-authoritative obs_mode (no [ml] required: ValueError fires
# before ML imports since manifest resolution was moved before torch/sb3 imports)
# ---------------------------------------------------------------------------


def test_behavior_clone_token_mismatch_raises_value_error(tmp_path):
    """behavior_clone(obs_mode='flat') on a token practicum → clear ValueError, not KeyError."""
    from locma.envs.distill import behavior_clone  # noqa: PLC0415

    p = tmp_path / "token.npz"
    _write_fake_token_practicum(p)
    with pytest.raises(ValueError, match="obs_mode"):
        behavior_clone(data=str(p), obs_mode="flat", epochs=1)


# ---------------------------------------------------------------------------
# behavior_clone token smoke tests (require [ml] extra)
# ---------------------------------------------------------------------------


def test_behavior_clone_token_no_obs_mode_follows_manifest(tmp_path):
    """behavior_clone(data=<token practicum>) with NO obs_mode trains successfully."""
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("torch")

    from locma.envs.distill import behavior_clone  # noqa: PLC0415

    p = tmp_path / "token.npz"
    _write_fake_token_practicum(p)
    out = str(tmp_path / "m.zip")

    # Key regression: omitting obs_mode must follow the manifest, not crash.
    result = behavior_clone(data=str(p), out=out, epochs=1, val_frac=0.5, verbose=0)

    assert isinstance(result, dict)
    assert result["out"] == out
    assert os.path.exists(out), "model zip was not saved"


def test_behavior_clone_token_returns_finite_agreement(tmp_path):
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("torch")
    gymnasium = pytest.importorskip("gymnasium")

    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.envs.distill import behavior_clone  # noqa: PLC0415

    p = tmp_path / "token.npz"
    _write_fake_token_practicum(p)
    out = str(tmp_path / "m.zip")

    # val_frac=0.5 ensures val games are non-empty (5 games × 0.5 = 2 val games).
    result = behavior_clone(
        data=str(p),
        out=out,
        epochs=1,
        obs_mode="token",
        val_frac=0.5,
        verbose=0,
    )

    assert isinstance(result, dict)
    assert "val_agreement" in result
    assert np.isfinite(result["val_agreement"]), f"val_agreement={result['val_agreement']!r}"
    assert result["out"] == out
    assert os.path.exists(out), "model zip was not saved"

    # Verify the saved model has a Dict observation space.
    loaded = MaskablePPO.load(out)
    assert isinstance(loaded.observation_space, gymnasium.spaces.Dict), (
        f"expected Dict obs space, got {type(loaded.observation_space)}"
    )
