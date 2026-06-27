"""Tests for the tokenized observation encoder (Task 1, PPO2).

TDD: all tests were written BEFORE the implementation and run to confirm RED.
Pure-numpy encode_battle_tokens tests are NOT gymnasium-gated so they run in
CI (which installs only the dev extra). Only the token_obs_space tests that
need gymnasium are gated inside their function bodies.
"""

from __future__ import annotations

import numpy as np
import pytest

from locma.core.views import BattleView, CardView  # noqa: E402
from locma.envs.encode import (  # noqa: E402
    MAX_TOKENS,
    N_TACTICAL,
    TOKEN_FEATS,
    encode_battle_tokens,
    token_obs_space,
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
    )


# ---------------------------------------------------------------------------
# (a) Shapes and dtypes
# ---------------------------------------------------------------------------


def test_shapes_and_dtypes():
    """encode_battle_tokens returns the 4 keys with correct shapes and float32 dtype."""
    card = _make_card(instance_id=1, card_id=1)
    view = _make_view(my_hand=(card,), my_board=(card,), op_board=(card,))
    result = encode_battle_tokens(view)

    assert set(result.keys()) == {"tokens", "card_ids", "token_mask", "scalars"}

    assert result["tokens"].shape == (MAX_TOKENS, TOKEN_FEATS)
    assert result["card_ids"].shape == (MAX_TOKENS,)
    assert result["token_mask"].shape == (MAX_TOKENS,)
    assert result["scalars"].shape == (N_TACTICAL,)

    for key in ("tokens", "card_ids", "token_mask", "scalars"):
        assert result[key].dtype == np.float32, f"{key} dtype should be float32"


# ---------------------------------------------------------------------------
# (b) Padding: pad rows zero, card_ids==0, token_mask==0, sum==real count
# ---------------------------------------------------------------------------


def test_padding_zero_rows_and_mask_sum():
    """Unused slots are all-zero with card_id==0 and token_mask==0."""
    # 1 hand card, 1 my_board card, 1 op_board card → 3 real tokens
    card = _make_card(instance_id=2, card_id=7)
    view = _make_view(my_hand=(card,), my_board=(card,), op_board=(card,))
    result = encode_battle_tokens(view)

    # token_mask sums to exactly the number of real cards
    assert result["token_mask"].sum() == 3.0

    # Real slots carry the correct card_id:
    #   slot 0 = my_hand[0], slot 8 = my_board[0], slot 14 = op_board[0]
    assert result["card_ids"][0] == 7.0, "hand slot 0 must carry card_id=7"
    assert result["card_ids"][8] == 7.0, "my_board slot 8 must carry card_id=7"
    assert result["card_ids"][14] == 7.0, "op_board slot 14 must carry card_id=7"

    # Pad slot indices: 1..7 (hand), 9..13 (my_board), 15..19 (op_board)
    pad_indices = list(range(1, 8)) + list(range(9, 14)) + list(range(15, 20))
    for i in pad_indices:
        assert np.all(result["tokens"][i] == 0.0), f"tokens[{i}] pad row should be zero"
        assert result["card_ids"][i] == 0.0, f"card_ids[{i}] pad should be 0"
        assert result["token_mask"][i] == 0.0, f"token_mask[{i}] pad should be 0"


def test_empty_board_all_pad():
    """With no cards at all, every slot is padded and mask sums to 0."""
    view = _make_view()
    result = encode_battle_tokens(view)

    assert result["token_mask"].sum() == 0.0
    assert np.all(result["tokens"] == 0.0)
    assert np.all(result["card_ids"] == 0.0)


def test_token_mask_matches_real_card_count():
    """token_mask sum equals len(my_hand) + len(my_board) + len(op_board)."""
    hand = tuple(_make_card(instance_id=i, card_id=i) for i in range(1, 4))  # 3
    board = tuple(_make_card(instance_id=i + 10, card_id=i + 10) for i in range(1, 5))  # 4
    op = tuple(_make_card(instance_id=i + 20, card_id=i + 20) for i in range(1, 3))  # 2
    view = _make_view(my_hand=hand, my_board=board, op_board=op)
    result = encode_battle_tokens(view)
    assert result["token_mask"].sum() == float(len(hand) + len(board) + len(op))


# ---------------------------------------------------------------------------
# (c) Tactical scalars: guard count, reachable_face_damage, lethal_available
# ---------------------------------------------------------------------------

# Scalar index map (per spec §1):
#  0=turn, 1=me_health, 2=op_health, 3=me_mana, 4=summonable_count,
#  5=op_hand_count, 6=my_board_count, 7=op_board_count, 8=opp_guard_count,
#  9=my_total_attack, 10=my_total_defense, 11=reachable_face_damage,
#  12=lethal_available
_IDX_GUARD = 8
_IDX_RFD = 11
_IDX_LETHAL = 12


def test_guard_creature_blocks_face_damage():
    """When an op-board creature has Guard, reachable_face_damage must be 0."""
    guard_card = _make_card(
        instance_id=10,
        card_id=50,
        type=0,
        attack=2,
        defense=5,
        abilities="---G--",  # G is index 3
        can_attack=False,
    )
    attacker = _make_card(
        instance_id=11,
        card_id=51,
        type=0,
        attack=8,
        defense=3,
        abilities="------",
        can_attack=True,
        has_attacked=False,
    )
    view = _make_view(my_board=(attacker,), op_board=(guard_card,), op_health=30)
    result = encode_battle_tokens(view)
    scalars = result["scalars"]

    assert scalars[_IDX_GUARD] == 1.0, "opp_guard_count should be 1"
    assert scalars[_IDX_RFD] == 0.0, "reachable_face_damage must be 0 when Guard is up"
    assert scalars[_IDX_LETHAL] == 0.0, "lethal_available must be 0 when face damage is 0"


def test_no_guard_reachable_face_damage_nonzero():
    """Without Guard, reachable_face_damage equals the ready attackers' total attack."""
    blocker = _make_card(
        instance_id=10,
        card_id=50,
        type=0,
        attack=2,
        defense=5,
        abilities="------",  # no Guard
        can_attack=False,
    )
    attacker = _make_card(
        instance_id=11,
        card_id=51,
        type=0,
        attack=8,
        defense=3,
        abilities="------",
        can_attack=True,
        has_attacked=False,
    )
    view = _make_view(my_board=(attacker,), op_board=(blocker,), op_health=30)
    result = encode_battle_tokens(view)
    scalars = result["scalars"]

    assert scalars[_IDX_GUARD] == 0.0, "opp_guard_count should be 0"
    assert scalars[_IDX_RFD] == 8.0, "reachable_face_damage should be attacker's attack=8"
    assert scalars[_IDX_LETHAL] == 0.0, "lethal_available should be 0 (8 < 30)"


def test_lethal_available_at_exact_boundary():
    """lethal_available is 1.0 when reachable_face_damage == op_health."""
    attacker = _make_card(
        instance_id=11,
        card_id=51,
        type=0,
        attack=10,
        defense=3,
        abilities="------",
        can_attack=True,
        has_attacked=False,
    )
    # op_health == attacker's attack == 10, no guard
    view = _make_view(my_board=(attacker,), op_board=(), op_health=10)
    result = encode_battle_tokens(view)
    scalars = result["scalars"]

    assert scalars[_IDX_RFD] == 10.0, "reachable_face_damage should be 10"
    assert scalars[_IDX_LETHAL] == 1.0, "lethal_available must be 1.0 when damage >= op_health"


def test_lethal_just_below_boundary():
    """lethal_available is 0.0 when reachable_face_damage < op_health by 1."""
    attacker = _make_card(
        instance_id=11,
        card_id=51,
        type=0,
        attack=9,
        defense=3,
        abilities="------",
        can_attack=True,
        has_attacked=False,
    )
    view = _make_view(my_board=(attacker,), op_board=(), op_health=10)
    result = encode_battle_tokens(view)
    scalars = result["scalars"]

    assert scalars[_IDX_RFD] == 9.0
    assert scalars[_IDX_LETHAL] == 0.0, "lethal_available must be 0 when damage < op_health"


def test_has_attacked_card_not_counted_in_face_damage():
    """A creature that has_attacked=True must not contribute to reachable_face_damage."""
    tired = _make_card(
        instance_id=11,
        card_id=51,
        type=0,
        attack=10,
        defense=3,
        abilities="------",
        can_attack=True,
        has_attacked=True,  # already attacked
    )
    view = _make_view(my_board=(tired,), op_board=(), op_health=5)
    result = encode_battle_tokens(view)
    scalars = result["scalars"]

    assert scalars[_IDX_RFD] == 0.0, "has_attacked creature must not count toward face damage"
    assert scalars[_IDX_LETHAL] == 0.0


def test_summonable_count_scalar():
    """summonable_count counts hand cards with cost <= me_mana."""
    cheap = _make_card(instance_id=1, card_id=1, cost=2)  # cost 2 <= mana 3 → counts
    mid = _make_card(instance_id=2, card_id=2, cost=3)  # cost 3 <= mana 3 → counts
    pricey = _make_card(instance_id=3, card_id=3, cost=4)  # cost 4 > mana 3 → no
    view = _make_view(my_hand=(cheap, mid, pricey), me_mana=3)
    result = encode_battle_tokens(view)
    # scalar index 4 = summonable_count
    assert result["scalars"][4] == 2.0


# ---------------------------------------------------------------------------
# (d) token_obs_space — spaces.Dict with correct shapes/dtypes
# ---------------------------------------------------------------------------


def test_token_obs_space_returns_dict_space():
    """token_obs_space() returns a gymnasium spaces.Dict."""
    gym = pytest.importorskip("gymnasium")
    space = token_obs_space()
    assert isinstance(space, gym.spaces.Dict)


def test_token_obs_space_shapes_and_dtypes_match_encoder():
    """spaces.Dict shapes and dtypes match what encode_battle_tokens produces."""
    pytest.importorskip("gymnasium")
    space = token_obs_space()
    card = _make_card(instance_id=1, card_id=1)
    view = _make_view(my_hand=(card,), my_board=(card,))
    encoded = encode_battle_tokens(view)

    for key in ("tokens", "card_ids", "token_mask", "scalars"):
        assert key in space.spaces, f"key '{key}' missing from obs space"
        assert encoded[key].shape == space.spaces[key].shape, (
            f"{key}: encoder shape {encoded[key].shape} != space shape {space.spaces[key].shape}"
        )
        assert encoded[key].dtype == space.spaces[key].dtype, (
            f"{key}: encoder dtype {encoded[key].dtype} != space dtype {space.spaces[key].dtype}"
        )
