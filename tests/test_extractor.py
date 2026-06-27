"""Tests for TokenSetExtractor (Task 2, PPO2).

TDD: all tests were written BEFORE the implementation and run to confirm RED.
torch + stable_baselines3 + gymnasium are importorskip-gated at module level.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

from locma.envs.encode import (  # noqa: E402
    MAX_TOKENS,
    N_TACTICAL,
    NUM_CARDS,
    TOKEN_FEATS,
    token_obs_space,
)
from locma.envs.extractor import TokenSetExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(B: int = 4, n_real: int = 10):
    """Build a random valid tensor batch with n_real real token slots.

    card_ids are float32 (SB3 batches Dict-obs keys as float32 from Box spaces);
    the extractor casts to long internally.
    """
    tokens = torch.zeros(B, MAX_TOKENS, TOKEN_FEATS)
    card_ids = torch.zeros(B, MAX_TOKENS)
    token_mask = torch.zeros(B, MAX_TOKENS)

    if n_real > 0:
        tokens[:, :n_real, :] = torch.randn(B, n_real, TOKEN_FEATS)
        card_ids[:, :n_real] = torch.randint(1, NUM_CARDS + 1, (B, n_real)).float()
        token_mask[:, :n_real] = 1.0

    scalars = torch.randn(B, N_TACTICAL)
    return {
        "tokens": tokens,
        "card_ids": card_ids,
        "token_mask": token_mask,
        "scalars": scalars,
    }


# ---------------------------------------------------------------------------
# (a) Forward shape + finiteness
# ---------------------------------------------------------------------------


def test_forward_shape_and_finite():
    """forward() returns (B, features_dim) with all finite values."""
    space = token_obs_space()
    extractor = TokenSetExtractor(space)
    obs = _make_batch(B=4, n_real=10)

    out = extractor(obs)

    assert out.shape == (4, 128), f"expected (4, 128), got {out.shape}"
    assert torch.isfinite(out).all(), "output contains non-finite values"


# ---------------------------------------------------------------------------
# (b) Gradients flow
# ---------------------------------------------------------------------------


def test_gradients_flow():
    """Gradients propagate back through at least the id embedding and proj weight."""
    space = token_obs_space()
    extractor = TokenSetExtractor(space)
    obs = _make_batch(B=4, n_real=10)

    out = extractor(obs)
    out.sum().backward()

    # Check that at least the embedding and proj weight have non-zero gradients.
    emb_grad = extractor.id_embed.weight.grad
    proj_grad = extractor.proj.weight.grad

    assert emb_grad is not None, "id_embed gradient is None"
    assert proj_grad is not None, "proj weight gradient is None"
    assert emb_grad.abs().sum() > 0, "id_embed gradient is all-zero"
    assert proj_grad.abs().sum() > 0, "proj weight gradient is all-zero"


# ---------------------------------------------------------------------------
# (c) Permutation / padding invariance (pool="cls", eval mode)
# ---------------------------------------------------------------------------


def test_permutation_padding_invariance():
    """Shuffling the 20 (token-row, card_id, mask) triples leaves CLS output identical.

    This proves the mask works and the model treats the token set as order-invariant
    (no positional encoding is added - by design).
    """
    torch.manual_seed(42)
    space = token_obs_space()
    extractor = TokenSetExtractor(space, pool="cls")
    extractor.eval()

    B, n_real = 1, 7  # 7 real tokens, 13 pads

    # Build original observation (real tokens in first n_real slots).
    obs = _make_batch(B=B, n_real=n_real)

    with torch.no_grad():
        out_orig = extractor(obs)

    # Build a random permutation of all 20 positions and apply to every key.
    perm = torch.randperm(MAX_TOKENS)
    obs_perm = {
        "tokens": obs["tokens"][:, perm, :],
        "card_ids": obs["card_ids"][:, perm],
        "token_mask": obs["token_mask"][:, perm],
        "scalars": obs["scalars"],  # scalars are not permuted
    }

    with torch.no_grad():
        out_perm = extractor(obs_perm)

    torch.testing.assert_close(out_orig, out_perm, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# (d) Smoke: pool="attn" and n_layers=1 produce finite (B, features_dim)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# (e) Input normalization: large-magnitude real-scale inputs stay finite
# ---------------------------------------------------------------------------


def test_large_magnitude_inputs_finite():
    """Input normalization must tame real-scale magnitudes (health~30, turn~50,
    attack/defense~12, board totals~60) and produce finite extractor output.

    This catches a regression where unnormalized raw values would cause NaN/Inf
    through the transformer (e.g. when LayerNorm is accidentally removed).
    """
    torch.manual_seed(0)
    space = token_obs_space()
    extractor = TokenSetExtractor(space)
    extractor.eval()

    B = 4
    # Build realistic worst-case magnitudes for the token features:
    # zone(3) + type(4) + cost/atk/def(3) + abilities(6) + ready(1) = 17
    tokens = torch.zeros(B, MAX_TOKENS, TOKEN_FEATS)
    tokens[:, :, 0] = 1.0  # zone one-hot
    tokens[:, :, 4] = 1.0  # type one-hot
    tokens[:, :, 5] = 8.0  # cost (max typical)
    tokens[:, :, 6] = 12.0  # attack (large)
    tokens[:, :, 7] = 12.0  # defense (large)

    card_ids = torch.randint(1, NUM_CARDS + 1, (B, MAX_TOKENS)).float()
    token_mask = torch.ones(B, MAX_TOKENS)

    # Tactical scalars at realistic maximum magnitudes:
    # turn, me_health, op_health, me_mana, summonable, op_hand, my_board,
    # op_board, guard_count, my_atk_total, my_def_total, rfd, lethal
    scalar_row = [50.0, 30.0, 30.0, 12.0, 8.0, 8.0, 6.0, 6.0, 6.0, 60.0, 60.0, 60.0, 1.0]
    scalars = torch.tensor([scalar_row] * B)

    obs = {
        "tokens": tokens,
        "card_ids": card_ids,
        "token_mask": token_mask,
        "scalars": scalars,
    }

    with torch.no_grad():
        out = extractor(obs)

    assert torch.isfinite(out).all(), (
        f"Output contains non-finite values with real-scale inputs; "
        f"min={out.min():.4f}, max={out.max():.4f}"
    )


def test_attn_pool_and_single_layer_smoke():
    """pool='attn' with n_layers=1 produces a finite (B, features_dim) tensor."""
    space = token_obs_space()
    extractor = TokenSetExtractor(space, pool="attn", n_layers=1)
    extractor.eval()

    obs = _make_batch(B=4, n_real=8)

    with torch.no_grad():
        out = extractor(obs)

    assert out.shape == (4, 128), f"expected (4, 128), got {out.shape}"
    assert torch.isfinite(out).all(), "attn-pool output contains non-finite values"
