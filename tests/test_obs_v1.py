from types import SimpleNamespace

import pytest

from locma.envs.encode import N_TACTICAL_V1, encode_battle_tokens, token_obs_space


def _card(card_id, atk, dfn, abilities="------", can_attack=False, has_attacked=False, cost=1):
    return SimpleNamespace(
        card_id=card_id,
        attack=atk,
        defense=dfn,
        abilities=abilities,
        can_attack=can_attack,
        has_attacked=has_attacked,
        cost=cost,
        type=0,
    )


def _view():
    # me: one 3/2 ready attacker; op: one 4/4 ready attacker with no guard.
    return SimpleNamespace(
        turn=5,
        me_health=10,
        op_health=20,
        me_mana=3,
        op_hand_count=4,
        my_hand=[_card(1, 0, 0, cost=2)],
        my_board=[_card(2, 3, 2, can_attack=True)],
        op_board=[_card(3, 4, 4, can_attack=True)],
    )


def test_v0_unchanged_length():
    s = encode_battle_tokens(_view(), variant="v0")["scalars"]
    assert s.shape[0] == 13


def test_v1_appends_five_symmetric_threat_scalars():
    s = encode_battle_tokens(_view(), variant="v1")["scalars"]
    assert s.shape[0] == N_TACTICAL_V1 == 18
    # [13]=my_guard_count=0, [14]=op_total_attack=4, [15]=op_reachable=4,
    # [16]=exposed_to_lethal (4>=10? no →0), [17]=card_advantage=(1+1)-(4+1)=-3
    assert s[13] == 0.0
    assert s[14] == 4.0
    assert s[15] == 4.0
    assert s[16] == 0.0
    assert s[17] == -3.0


def test_token_obs_space_v1_scalar_shape():
    # Gated per-function (not module-level) so the numpy-only encode tests
    # above still run on the lean CI job, matching test_encode_tokens.py.
    pytest.importorskip("gymnasium")
    sp = token_obs_space(variant="v1")
    assert sp["scalars"].shape == (18,)


def test_v1_op_reachable_includes_creatures_that_already_attacked():
    """start_turn refreshes every op-board creature at the start of the
    opponent's turn, so a creature that already attacked THIS turn (my
    decision point) will be ready again on the opponent's next turn -- it
    must still count toward op_reachable. Filtering by current readiness
    (the old, buggy behavior) would drop this creature and undercount."""
    view = SimpleNamespace(
        turn=5,
        me_health=10,
        op_health=20,
        me_mana=3,
        op_hand_count=4,
        my_hand=[],
        my_board=[],
        op_board=[_card(3, 4, 4, can_attack=False, has_attacked=True)],
    )
    s = encode_battle_tokens(view, variant="v1")["scalars"]
    assert s[15] == 4.0  # op_reachable includes the already-attacked creature


def test_v1_op_reachable_zero_with_my_guard():
    view = SimpleNamespace(
        turn=5,
        me_health=10,
        op_health=20,
        me_mana=3,
        op_hand_count=4,
        my_hand=[],
        my_board=[_card(2, 1, 3, abilities="---G--")],
        op_board=[_card(3, 4, 4, can_attack=True)],
    )
    s = encode_battle_tokens(view, variant="v1")["scalars"]
    assert s[15] == 0.0  # my_guard_count > 0 gates op_reachable to 0
