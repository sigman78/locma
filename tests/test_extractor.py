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
    TOKEN_FEATS_FX,
    token_obs_space,
)
from locma.envs.extractor import SlimTokenExtractor, TokenSetExtractor  # noqa: E402

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


def test_default_dropout_pinned():
    """dropout=0.1 is empirically load-bearing: the 2026-07-02 N-battery showed
    removing it regresses -0.028 paired avg-hard3 at the tuned recipe (despite
    the known PPO ratio-noise caveat). Pin the default so it only changes
    deliberately, with a fresh paired ceiling-eval."""
    space = token_obs_space()
    extractor = TokenSetExtractor(space)
    encoder_layer = extractor.transformer.layers[0]
    assert encoder_layer.dropout.p == 0.1


def test_forward_shape_and_finite():
    """forward() returns (B, features_dim) with all finite values."""
    space = token_obs_space()
    extractor = TokenSetExtractor(space)
    obs = _make_batch(B=4, n_real=10)

    out = extractor(obs)

    assert out.shape == (4, 256), f"expected (4, 256), got {out.shape}"
    assert torch.isfinite(out).all(), "output contains non-finite values"


def test_slim_extractor_forward_and_slots():
    """E29 slim arm: SlimTokenExtractor forwards to (B, features_dim) and its
    slot_encoder emits (B, MAX_TOKENS, d_model) — the pointer gather source,
    transformer-free."""
    space = token_obs_space()
    ex = SlimTokenExtractor(space, d_model=64)
    assert not hasattr(ex, "transformer")  # transformer-free by construction
    obs = _make_batch(B=6, n_real=9)

    slots = ex.slot_encoder(torch.cat([obs["tokens"], ex.id_embed(obs["card_ids"].long())], dim=-1))
    assert slots.shape == (6, MAX_TOKENS, 64)

    out = ex(obs)
    assert out.shape == (6, 256)
    assert torch.isfinite(out).all()


def test_slim_extractor_all_pad_row_finite():
    """A fully-padded row (n_real=0) must not NaN via the max-pool guard."""
    space = token_obs_space()
    ex = SlimTokenExtractor(space)
    obs = _make_batch(B=3, n_real=0)  # every slot is pad
    out = ex(obs)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# E42 public-history branch: opt-in via a "hist" key on the obs space.
# ---------------------------------------------------------------------------


def _make_fx_batch(B: int = 4, n_real: int = 10):
    """Like ``_make_batch`` but at the fx/fxh token width (20, not 17)."""
    tokens = torch.zeros(B, MAX_TOKENS, TOKEN_FEATS_FX)
    card_ids = torch.zeros(B, MAX_TOKENS)
    token_mask = torch.zeros(B, MAX_TOKENS)

    if n_real > 0:
        tokens[:, :n_real, :] = torch.randn(B, n_real, TOKEN_FEATS_FX)
        card_ids[:, :n_real] = torch.randint(1, NUM_CARDS + 1, (B, n_real)).float()
        token_mask[:, :n_real] = 1.0

    scalars = torch.randn(B, N_TACTICAL)
    return {
        "tokens": tokens,
        "card_ids": card_ids,
        "token_mask": token_mask,
        "scalars": scalars,
    }


# Hard-coded so any accidental new parameter on the no-hist path fails loudly.
_SLIM_STATE_DICT_KEYS_NO_HIST = {
    "id_embed.weight",
    "slot_encoder.proj.weight",
    "slot_encoder.proj.bias",
    "slot_encoder.token_ln.weight",
    "slot_encoder.token_ln.bias",
    "slot_encoder.pos_embed",
    "scalar_mlp.0.weight",
    "scalar_mlp.0.bias",
    "scalar_mlp.1.weight",
    "scalar_mlp.1.bias",
    "head.0.weight",
    "head.0.bias",
}


def test_slim_extractor_no_hist_key_state_dict_unchanged():
    """Without a "hist" obs key, SlimTokenExtractor gets no new parameters —
    old fx/v0 checkpoints must keep loading unchanged (E42)."""
    space = token_obs_space("fx")
    assert "hist" not in space.spaces
    ex = SlimTokenExtractor(space)
    assert ex.hist_mlp is None
    assert set(ex.state_dict().keys()) == _SLIM_STATE_DICT_KEYS_NO_HIST


def test_slim_extractor_hist_branch_present_and_zero_init():
    """With a "hist" obs key (variant "fxh"), a LayerNorm->Linear->ReLU->Linear
    branch appears, added to the scalar-branch output before `head`. The last
    Linear's weight AND bias are zero-initialized, so at construction time the
    branch is a no-op regardless of the "hist" input (E42's warm-start
    contract: a freshly widened fxh net computes EXACTLY the same features as
    the hist-less net)."""
    space = token_obs_space("fxh")
    assert "hist" in space.spaces
    ex = SlimTokenExtractor(space)
    assert isinstance(ex.hist_mlp, torch.nn.Sequential)
    last = ex.hist_mlp[-1]
    assert isinstance(last, torch.nn.Linear)
    assert torch.all(last.weight == 0)
    assert torch.all(last.bias == 0)

    n_hist = int(space["hist"].shape[0])
    obs = _make_fx_batch(B=4, n_real=10)
    obs["hist"] = torch.zeros(4, n_hist)
    out_zero_hist = ex(obs)

    obs_rand_hist = dict(obs)
    obs_rand_hist["hist"] = torch.randn(4, n_hist)
    out_rand_hist = ex(obs_rand_hist)

    # Zero-init last layer -> hist_mlp(anything) == 0 at init, so the output is
    # identical regardless of the hist input.
    assert torch.equal(out_zero_hist, out_rand_hist)


def test_slim_extractor_hist_branch_matches_no_hist_extractor_at_init():
    """A freshly-built fxh extractor and a freshly-built fx (no-hist) extractor
    with the SAME non-hist weights produce byte-identical output at init —
    the warm-start contract this branch exists to satisfy."""
    torch.manual_seed(0)
    space_fx = token_obs_space("fx")
    space_fxh = token_obs_space("fxh")
    ex_fx = SlimTokenExtractor(space_fx)
    ex_fxh = SlimTokenExtractor(space_fxh)
    # Copy the shared (non-hist) parameters from ex_fx onto ex_fxh, as a
    # warm-start would via load_state_dict(strict=False).
    missing, unexpected = ex_fxh.load_state_dict(ex_fx.state_dict(), strict=False)
    assert unexpected == []
    assert set(missing) == {
        "hist_mlp.0.weight",
        "hist_mlp.0.bias",
        "hist_mlp.1.weight",
        "hist_mlp.1.bias",
        "hist_mlp.3.weight",
        "hist_mlp.3.bias",
    }

    obs = _make_fx_batch(B=4, n_real=10)
    n_hist = int(space_fxh["hist"].shape[0])
    obs_fxh = dict(obs)
    obs_fxh["hist"] = torch.randn(4, n_hist)

    ex_fx.eval()
    ex_fxh.eval()
    with torch.no_grad():
        out_fx = ex_fx(obs)
        out_fxh = ex_fxh(obs_fxh)
    assert torch.equal(out_fx, out_fxh)


def test_feature_ln_optin(monkeypatch):
    """E29 conditioned-trunk lever: feature_ln adds a LayerNorm on the tower
    input (default OFF = no such module, byte-identical to e28c/e28p), and the
    conditioned output is per-sample zero-mean/unit-var before the affine."""
    space = token_obs_space()
    obs = _make_batch(B=8, n_real=10)

    off = TokenSetExtractor(space)
    assert off.out_ln is None  # default OFF: no extra module -> old artifacts load

    on = TokenSetExtractor(space, feature_ln=True)
    assert isinstance(on.out_ln, torch.nn.LayerNorm)
    out = on(obs)
    assert out.shape == (8, 256)
    assert torch.isfinite(out).all()
    # LayerNorm affine starts at gamma=1/beta=0, so the raw output is normalized.
    assert out.mean(dim=-1).abs().max() < 1e-4


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
# (c) Slot-sensitivity test (replaces permutation-invariance test)
# ---------------------------------------------------------------------------


def test_slot_sensitivity():
    """Swapping two DISTINCT cards between slots produces different output features.

    The action space is slot-indexed (Summon→1+s, Use→9+s*13+tc,
    Attack→113+a*7+tc), so the policy must distinguish which card occupies which
    slot. A permutation-invariant pooled encoder would return identical features
    after the swap. The slot-addressable encoder (positional embedding + per-slot
    flatten) must NOT be invariant — this test asserts that property.
    """
    torch.manual_seed(42)
    space = token_obs_space()
    extractor = TokenSetExtractor(space)
    extractor.eval()

    B = 1

    # Two distinct cards: different card_ids and different random feature rows.
    card_a_feats = torch.randn(TOKEN_FEATS)
    card_b_feats = torch.randn(TOKEN_FEATS)
    card_a_id = 1.0
    card_b_id = 2.0

    def _obs_with_cards(a_slot: int, b_slot: int):
        tokens = torch.zeros(B, MAX_TOKENS, TOKEN_FEATS)
        card_ids = torch.zeros(B, MAX_TOKENS)
        token_mask = torch.zeros(B, MAX_TOKENS)
        tokens[0, a_slot] = card_a_feats
        card_ids[0, a_slot] = card_a_id
        token_mask[0, a_slot] = 1.0
        tokens[0, b_slot] = card_b_feats
        card_ids[0, b_slot] = card_b_id
        token_mask[0, b_slot] = 1.0
        return {
            "tokens": tokens,
            "card_ids": card_ids,
            "token_mask": token_mask,
            "scalars": torch.zeros(B, N_TACTICAL),
        }

    obs_a = _obs_with_cards(a_slot=0, b_slot=1)  # A in slot 0, B in slot 1
    obs_b = _obs_with_cards(a_slot=1, b_slot=0)  # A in slot 1, B in slot 0 (swapped)

    with torch.no_grad():
        out_a = extractor(obs_a)
        out_b = extractor(obs_b)

    assert not torch.allclose(out_a, out_b, atol=1e-4), (
        "Extractor output is identical after swapping distinct cards between slots — "
        "the encoder is still permutation-invariant, but the slot-indexed action "
        "space requires slot-sensitive features."
    )


# ---------------------------------------------------------------------------
# (d) All-pad guard: no NaN when all tokens are padding
# ---------------------------------------------------------------------------


def test_all_pad_guard_finite():
    """An input where every slot is padding returns a finite (B, features_dim) tensor.

    The all-pad guard in forward() unmasks all positions for fully-padded rows so
    the transformer's attention does not produce NaN (attending over zero keys).
    """
    space = token_obs_space()
    extractor = TokenSetExtractor(space)
    extractor.eval()

    B = 2
    obs = {
        "tokens": torch.zeros(B, MAX_TOKENS, TOKEN_FEATS),
        "card_ids": torch.zeros(B, MAX_TOKENS),
        "token_mask": torch.zeros(B, MAX_TOKENS),  # all pad — every slot masked
        "scalars": torch.zeros(B, N_TACTICAL),
    }

    with torch.no_grad():
        out = extractor(obs)

    assert out.shape == (B, 256), f"expected ({B}, 256), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf for all-pad input"


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
