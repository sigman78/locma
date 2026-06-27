# tests/test_distill_token.py
"""Token-obs mode distillation tests.

Requires the [ml] extra (sb3_contrib + torch); skipped otherwise via module-level
importorskip (following the pattern of test_training_token.py).
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


def _write_fake_token_practicum(path, n=20, n_games=5):
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
    with open(_manifest_path(str(path)), "w") as f:
        json.dump(_TOKEN_MANIFEST, f)


# ---------------------------------------------------------------------------
# load_practicum tests (pure numpy — no ML deps required)
# ---------------------------------------------------------------------------


def test_load_practicum_token_returns_four_keys(tmp_path):
    from locma.envs.distill import load_practicum  # noqa: PLC0415

    p = tmp_path / "token.npz"
    _write_fake_token_practicum(p)
    arrays, manifest = load_practicum(str(p))
    assert manifest["obs_mode"] == "token"
    for key in ("obs_tokens", "obs_card_ids", "obs_token_mask", "obs_scalars"):
        assert key in arrays, f"missing key {key!r}"
    assert arrays["obs_tokens"].shape == (20, MAX_TOKENS, TOKEN_FEATS)
    assert arrays["obs_scalars"].shape == (20, N_TACTICAL)


def test_load_practicum_token_rejects_wrong_token_feats(tmp_path):
    from locma.envs.distill import load_practicum  # noqa: PLC0415

    p = tmp_path / "bad.npz"
    _write_fake_token_practicum(p)
    bad_manifest = dict(_TOKEN_MANIFEST, token_feats=99)
    with open(_manifest_path(str(p)), "w") as f:
        json.dump(bad_manifest, f)
    with pytest.raises(ValueError, match="layout"):
        load_practicum(str(p))


# ---------------------------------------------------------------------------
# behavior_clone token smoke test (requires [ml] extra)
# ---------------------------------------------------------------------------

pytest.importorskip("sb3_contrib")
pytest.importorskip("torch")
gymnasium = pytest.importorskip("gymnasium")

from sb3_contrib import MaskablePPO  # noqa: E402

from locma.envs.distill import behavior_clone  # noqa: E402


def test_behavior_clone_token_returns_finite_agreement(tmp_path):
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
