"""Tests for the "fxh" token-obs variant (E42 public-history features).

fxh = fx tokens (20-wide) + v0 scalars (13) + a new "hist" key (25 raw-count
public-history features: own remaining deck + the opponent's played cards and
turn-end mana/hand log). See docs/e42-public-history-plan.md, "Plumbing".
"""

from __future__ import annotations

import numpy as np
import pytest

from locma.core.views import BattleView, CardView
from locma.envs.encode import (
    MAX_TOKENS,
    N_HIST,
    N_TACTICAL,
    N_TACTICAL_V1,
    TOKEN_FEATS_FX,
    encode_battle_tokens,
    encode_battle_tokens_batch,
    hist_features,
    token_obs_space,
    token_variant_for_space,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_card(
    instance_id: int = 1,
    card_id: int = 1,
    type: int = 0,
    cost: int = 1,
    attack: int = 2,
    defense: int = 3,
    abilities: str = "------",
    can_attack: bool = False,
    has_attacked: bool = False,
) -> CardView:
    return CardView(
        instance_id=instance_id,
        card_id=card_id,
        type=type,
        cost=cost,
        attack=attack,
        defense=defense,
        abilities=abilities,
        can_attack=can_attack,
        has_attacked=has_attacked,
    )


def _make_view(
    my_hand=(),
    my_board=(),
    op_board=(),
    turn: int = 1,
    me_health: int = 30,
    me_mana: int = 5,
    op_health: int = 30,
    op_hand_count: int = 3,
    my_deck_count: int = 0,
    op_deck_count: int = 0,
    my_deck_cards: tuple = (),
    op_played: tuple = (),
    op_turn_log: tuple = (),
) -> BattleView:
    return BattleView(
        turn=turn,
        me_health=me_health,
        me_mana=me_mana,
        op_health=op_health,
        op_hand_count=op_hand_count,
        my_hand=my_hand,
        my_board=my_board,
        op_board=op_board,
        my_deck_count=my_deck_count,
        op_deck_count=op_deck_count,
        my_deck_cards=my_deck_cards,
        op_played=op_played,
        op_turn_log=op_turn_log,
    )


# Real card ids from the LOCM 1.2 cardlist with known (type, cost), picked by
# hand so the expected hist vector below can be hand-computed:
#   1   creature cost 1  (bucket 0-2)
#   9   creature cost 3  (bucket 3-4)
#   117 green item cost 1 (bucket 0-2)
#   151 red item cost 5   (bucket 5-6)
#   153 blue item cost 2  (bucket 0-2)
_CREATURE_C1 = 1
_CREATURE_C3 = 9
_GREEN_C1 = 117
_RED_C5 = 151
_BLUE_C2 = 153


def _history_view() -> BattleView:
    return _make_view(
        my_deck_count=4,
        my_deck_cards=(_CREATURE_C1, _CREATURE_C3, _GREEN_C1, _RED_C5),
        op_deck_count=15,
        op_played=(_BLUE_C2, _CREATURE_C3, _RED_C5),
        op_turn_log=((3, 4), (1, 2), (0, 5)),
    )


# ---------------------------------------------------------------------------
# hist_features: shape/dtype + hand-computed values
# ---------------------------------------------------------------------------


def test_hist_features_shape_and_dtype():
    h = hist_features(_make_view())
    assert h.shape == (N_HIST,)
    assert h.dtype == np.float32


def test_hist_features_all_zero_for_empty_history():
    h = hist_features(_make_view())
    np.testing.assert_array_equal(h, np.zeros(N_HIST, dtype=np.float32))


def test_hist_features_hand_computed_values():
    h = hist_features(_history_view())
    expected = np.array(
        [
            4.0,  # 0  my_deck_count
            2.0,
            1.0,
            1.0,
            0.0,  # 1-4  own deck by type: creature, green, red, blue
            2.0,
            1.0,
            1.0,
            0.0,  # 5-8  own deck by cost bucket: 0-2,3-4,5-6,7+
            2.5,  # 9  own deck mean cost = (1+3+1+5)/4
            15.0,  # 10 op_deck_count
            3.0,  # 11 n opponent cards played
            1.0,
            0.0,
            1.0,
            1.0,  # 12-15 opponent played by type: creature, green, red, blue
            1.0,
            1.0,
            1.0,
            0.0,  # 16-19 opponent played by cost bucket
            10.0 / 3.0,  # 20 opponent played mean cost = (2+3+5)/3
            0.0,  # 21 mana_left at last op turn end
            4.0 / 3.0,  # 22 mean op mana_left = (3+1+0)/3
            1.0 / 3.0,  # 23 sandbag rate: only (3,4) qualifies (mana>=2 and hand>0)
            0.0,  # 24 last turn was NOT a sandbag turn ((0,5): mana<2)
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(h, expected, rtol=1e-6, atol=1e-6)


def test_hist_features_last_turn_sandbag_flag_true():
    view = _make_view(op_turn_log=((5, 1), (2, 3)))
    h = hist_features(view)
    assert h[21] == 2.0  # last mana_left
    assert h[24] == 1.0  # last turn WAS a sandbag turn (2>=2 and 3>0)
    assert h[23] == 1.0  # both entries qualify


# ---------------------------------------------------------------------------
# encode_battle_tokens("fxh")
# ---------------------------------------------------------------------------


def test_fxh_shapes_and_scalars():
    obs = encode_battle_tokens(_history_view(), variant="fxh")
    assert set(obs.keys()) == {"tokens", "card_ids", "token_mask", "scalars", "hist"}
    assert obs["tokens"].shape == (MAX_TOKENS, TOKEN_FEATS_FX)
    assert obs["scalars"].shape == (N_TACTICAL,)
    assert obs["hist"].shape == (N_HIST,)
    assert obs["hist"].dtype == np.float32


def test_fxh_hist_matches_hist_features():
    view = _history_view()
    obs = encode_battle_tokens(view, variant="fxh")
    np.testing.assert_array_equal(obs["hist"], hist_features(view))


def test_fxh_tokens_and_scalars_byte_identical_to_fx():
    """fxh's tokens/scalars are exactly fx's — hist is purely additive."""
    card = _make_card(instance_id=1, card_id=_CREATURE_C1)
    view = _make_view(
        my_hand=(card,),
        my_board=(_make_card(instance_id=2, card_id=_CREATURE_C3, can_attack=True),),
        op_board=(_make_card(instance_id=3, card_id=_RED_C5, abilities="---G--"),),
        my_deck_cards=(_CREATURE_C1,),
        op_played=(_RED_C5,),
        op_turn_log=((2, 1),),
    )
    fx = encode_battle_tokens(view, variant="fx")
    fxh = encode_battle_tokens(view, variant="fxh")
    np.testing.assert_array_equal(fxh["tokens"], fx["tokens"])
    np.testing.assert_array_equal(fxh["card_ids"], fx["card_ids"])
    np.testing.assert_array_equal(fxh["token_mask"], fx["token_mask"])
    np.testing.assert_array_equal(fxh["scalars"], fx["scalars"])
    assert "hist" not in fx


def test_v0_and_fx_unaffected_by_fxh_addition():
    """fx/v0 outputs are byte-identical to what they were before fxh existed."""
    view = _history_view()
    v0 = encode_battle_tokens(view, variant="v0")
    fx = encode_battle_tokens(view, variant="fx")
    assert set(v0.keys()) == {"tokens", "card_ids", "token_mask", "scalars"}
    assert set(fx.keys()) == {"tokens", "card_ids", "token_mask", "scalars"}
    assert v0["tokens"].shape[1] != fx["tokens"].shape[1]  # sanity: fx is wider


# ---------------------------------------------------------------------------
# token_obs_space / token_variant_for_space
# ---------------------------------------------------------------------------


def test_fxh_obs_space_has_hist_key():
    pytest.importorskip("gymnasium")
    space = token_obs_space("fxh")
    assert "hist" in space.spaces
    assert space["hist"].shape == (N_HIST,)
    assert space["hist"].dtype == np.float32
    assert space["tokens"].shape == (MAX_TOKENS, TOKEN_FEATS_FX)
    assert space["scalars"].shape == (N_TACTICAL,)


def test_token_variant_for_space_detects_fxh():
    pytest.importorskip("gymnasium")
    space = token_obs_space("fxh")
    assert token_variant_for_space(space) == "fxh"


def test_token_variant_for_space_detects_fxh_plain_dict():
    """token_variant_for_space must also work on a plain dict (no .spaces)."""
    plain = {
        "tokens": np.zeros((MAX_TOKENS, TOKEN_FEATS_FX), dtype=np.float32),
        "card_ids": np.zeros((MAX_TOKENS,), dtype=np.float32),
        "token_mask": np.zeros((MAX_TOKENS,), dtype=np.float32),
        "scalars": np.zeros((N_TACTICAL,), dtype=np.float32),
        "hist": np.zeros((N_HIST,), dtype=np.float32),
    }
    assert token_variant_for_space(plain) == "fxh"


def test_token_variant_for_space_other_variants_unchanged():
    pytest.importorskip("gymnasium")
    assert token_variant_for_space(token_obs_space("v0")) == "v0"
    assert token_variant_for_space(token_obs_space("v1")) == "v1"
    assert token_variant_for_space(token_obs_space("fx")) == "fx"
    assert token_obs_space("v1")["scalars"].shape == (N_TACTICAL_V1,)


# ---------------------------------------------------------------------------
# encode_battle_tokens_batch("fxh")
# ---------------------------------------------------------------------------


def test_fxh_batch_equals_stacked_singles():
    views = [
        _history_view(),
        _make_view(),  # all-empty history
        _make_view(
            my_hand=(_make_card(instance_id=1, card_id=_GREEN_C1),),
            my_deck_cards=(_BLUE_C2, _RED_C5),
            op_played=(_CREATURE_C1,),
            op_turn_log=((0, 0), (4, 2)),
        ),
    ]
    batch = encode_battle_tokens_batch(views, "fxh")
    assert set(batch.keys()) == {"tokens", "card_ids", "token_mask", "scalars", "hist"}
    ref = {
        k: np.stack([encode_battle_tokens(v, "fxh")[k] for v in views])
        for k in ("tokens", "card_ids", "token_mask", "scalars", "hist")
    }
    for k, value in ref.items():
        np.testing.assert_array_equal(batch[k], value)


def test_fxh_batch_empty_views_list():
    batch = encode_battle_tokens_batch([], "fxh")
    assert batch["hist"].shape == (0, N_HIST)


def test_batch_fx_v0_v1_unaffected_by_fxh():
    """Non-fxh variants of the batch encoder never carry a "hist" key."""
    views = [_make_view(), _history_view()]
    for variant in ("v0", "v1", "fx"):
        batch = encode_battle_tokens_batch(views, variant)
        assert "hist" not in batch
