# tests/test_distill_token_load.py
"""Pure-numpy load_practicum tests for token mode — run without the [ml] extra.

The behavior_clone smoke tests that need sb3_contrib/torch stay in
test_distill_token.py (with importorskip inside each function).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from locma.envs.encode import ACTION_SIZE, MAX_TOKENS, N_TACTICAL, TOKEN_FEATS, TOKEN_FEATS_FX
from locma.envs.practicum import _manifest_path

_TOKEN_MANIFEST = {
    "obs_mode": "token",
    "max_tokens": MAX_TOKENS,
    "token_feats": TOKEN_FEATS,
    "n_tactical": N_TACTICAL,
    "action_size": ACTION_SIZE,
}


def _write_fake_token_practicum(path, n: int = 20, n_games: int = 5, tok_feats=TOKEN_FEATS) -> None:
    rng = np.random.default_rng(42)
    game_id = np.repeat(np.arange(n_games, dtype=np.int32), n // n_games)
    np.savez(
        path,
        obs_tokens=rng.standard_normal((n, MAX_TOKENS, tok_feats)).astype(np.float32),
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
# load_practicum: basic round-trip
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


# ---------------------------------------------------------------------------
# load_practicum: layout mismatch (Item 5 — parametrized over all four fields)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("token_feats", 99),
        ("max_tokens", 99),
        ("n_tactical", 99),
        ("action_size", 99),
    ],
)
def test_load_practicum_token_rejects_wrong_layout(tmp_path, field, bad_value):
    from locma.envs.distill import load_practicum  # noqa: PLC0415

    p = tmp_path / "bad.npz"
    _write_fake_token_practicum(p)
    bad_manifest = dict(_TOKEN_MANIFEST, **{field: bad_value})
    with open(_manifest_path(str(p)), "w", encoding="utf-8") as f:
        json.dump(bad_manifest, f)
    with pytest.raises(ValueError, match="layout"):
        load_practicum(str(p))


# ---------------------------------------------------------------------------
# load_practicum: token-fx variant (20-wide tokens, E29 slim re-probe path)
# ---------------------------------------------------------------------------

_FX_MANIFEST = dict(_TOKEN_MANIFEST, obs_mode="token-fx", token_feats=TOKEN_FEATS_FX)


def test_load_practicum_token_fx_returns_wide_tokens(tmp_path):
    from locma.envs.distill import load_practicum  # noqa: PLC0415

    p = tmp_path / "fx.npz"
    _write_fake_token_practicum(p, tok_feats=TOKEN_FEATS_FX)
    with open(_manifest_path(str(p)), "w", encoding="utf-8") as f:
        json.dump(_FX_MANIFEST, f)
    arrays, manifest = load_practicum(str(p))
    assert manifest["obs_mode"] == "token-fx"
    assert arrays["obs_tokens"].shape == (20, MAX_TOKENS, TOKEN_FEATS_FX)


def test_load_practicum_token_fx_rejects_v0_width(tmp_path):
    """A token-fx manifest carrying the plain 17-wide token_feats must be
    rejected — feeding a v0 practicum to an fx net is the exact bug this guards."""
    from locma.envs.distill import load_practicum  # noqa: PLC0415

    p = tmp_path / "fx_bad.npz"
    _write_fake_token_practicum(p, tok_feats=TOKEN_FEATS_FX)
    bad_manifest = dict(_FX_MANIFEST, token_feats=TOKEN_FEATS)
    with open(_manifest_path(str(p)), "w", encoding="utf-8") as f:
        json.dump(bad_manifest, f)
    with pytest.raises(ValueError, match="layout"):
        load_practicum(str(p))
