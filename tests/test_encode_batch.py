"""Tests for ``encode_battle_tokens_batch`` — the batched token encoder (E41).

The contract is byte-identity with stacking the per-view encoder. Profiling put
``encode_battle_tokens`` at 20.4 us/view — 24% of a batch-64 critic call and the
largest single line in the ``rbeam`` profile, larger than every
``torch.nn.linear`` combined — so the batched form is on the hot path for every
search policy, and a silent divergence would change play everywhere.

Engine + numpy only; no [ml] extra needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from locma.core.engine import make_battle_view, run_game
from locma.core.state import Phase
from locma.envs.encode import (
    MAX_TOKENS,
    N_TACTICAL,
    N_TACTICAL_V1,
    TOKEN_FEATS,
    TOKEN_FEATS_FX,
    encode_battle_tokens,
    encode_battle_tokens_batch,
)
from locma.policies.registry import make_policy

KEYS = ("tokens", "card_ids", "token_mask", "scalars")
VARIANTS = ("v0", "v1", "fx")


def _views(n=80, seed=606):
    """Real BattleViews spanning early/mid/late boards from played-out games."""
    out = []

    def hook(seat, action, gs):
        if gs.phase == Phase.BATTLE and len(out) < n:
            out.append(make_battle_view(gs))

    s = seed
    while len(out) < n:
        run_game(make_policy("greedy"), make_policy("scripted"), s, on_pre_step=hook)
        s += 1
    return out[:n]


def _reference(views, variant):
    return {k: np.stack([encode_battle_tokens(v, variant)[k] for v in views]) for k in KEYS}


@pytest.mark.parametrize("variant", VARIANTS)
def test_byte_identical_to_stacked_per_view(variant):
    views = _views()
    ref = _reference(views, variant)
    got = encode_battle_tokens_batch(views, variant)
    assert set(got) == set(KEYS)
    for k in KEYS:
        assert got[k].shape == ref[k].shape, f"{variant}/{k} shape"
        assert got[k].dtype == ref[k].dtype, f"{variant}/{k} dtype"
        assert np.array_equal(got[k], ref[k]), f"{variant}/{k} not byte-identical"


@pytest.mark.parametrize("variant", VARIANTS)
def test_shapes_match_the_declared_constants(variant):
    views = _views(12)
    got = encode_battle_tokens_batch(views, variant)
    feats = TOKEN_FEATS_FX if variant == "fx" else TOKEN_FEATS
    n_scalar = N_TACTICAL_V1 if variant == "v1" else N_TACTICAL
    assert got["tokens"].shape == (12, MAX_TOKENS, feats)
    assert got["card_ids"].shape == (12, MAX_TOKENS)
    assert got["token_mask"].shape == (12, MAX_TOKENS)
    assert got["scalars"].shape == (12, n_scalar)


def test_single_view_batch_matches():
    """B=1 is the degenerate case the scatter has to get right."""
    v = _views(1)
    for variant in VARIANTS:
        got = encode_battle_tokens_batch(v, variant)
        ref = encode_battle_tokens(v[0], variant)
        for k in KEYS:
            assert np.array_equal(got[k][0], ref[k]), f"{variant}/{k}"


def test_empty_batch_returns_correct_shapes():
    for variant in VARIANTS:
        got = encode_battle_tokens_batch([], variant)
        feats = TOKEN_FEATS_FX if variant == "fx" else TOKEN_FEATS
        assert got["tokens"].shape == (0, MAX_TOKENS, feats)
        assert got["scalars"].shape[0] == 0


def test_batch_order_is_preserved():
    """Row i must be view i — a scatter bug could permute silently."""
    views = _views(24)
    got = encode_battle_tokens_batch(views, "fx")
    for i, v in enumerate(views):
        ref = encode_battle_tokens(v, "fx")
        assert np.array_equal(got["tokens"][i], ref["tokens"]), f"row {i} mismatched"


def test_reordering_inputs_reorders_outputs_identically():
    views = _views(20)
    perm = [19, 0, 7, 3, 11, 2, 15, 8, 1, 4, 18, 5, 9, 12, 6, 17, 10, 13, 16, 14]
    a = encode_battle_tokens_batch([views[i] for i in perm], "fx")
    b = encode_battle_tokens_batch(views, "fx")
    for k in KEYS:
        assert np.array_equal(a[k], b[k][perm]), f"{k} not permutation-equivariant"


def test_fx_effect_columns_only_on_hand_slots():
    """Board slots must keep zeros in the fx columns (effects fire on play)."""
    views = _views(40)
    got = encode_battle_tokens_batch(views, "fx")
    board = got["tokens"][:, 8:, TOKEN_FEATS:]
    assert np.all(board == 0.0), "fx columns leaked onto board slots"


def test_pad_slots_are_all_zero():
    views = _views(40)
    got = encode_battle_tokens_batch(views, "fx")
    pads = got["token_mask"] == 0.0
    assert np.all(got["tokens"][pads] == 0.0), "pad token rows must be zero"
    assert np.all(got["card_ids"][pads] == 0.0), "pad card_ids must be zero"


def test_does_not_mutate_views():
    views = _views(20)
    before = [
        (
            len(v.my_hand),
            len(v.my_board),
            len(v.op_board),
            v.me_health,
            v.op_health,
            v.me_mana,
        )
        for v in views
    ]
    encode_battle_tokens_batch(views, "fx")
    after = [
        (
            len(v.my_hand),
            len(v.my_board),
            len(v.op_board),
            v.me_health,
            v.op_health,
            v.me_mana,
        )
        for v in views
    ]
    assert before == after
