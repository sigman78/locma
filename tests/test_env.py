from __future__ import annotations

import pytest

gym = pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402

from locma.envs.battle_env import BattleEnv  # noqa: E402
from locma.envs.encode import OBS_SIZE, encode_battle  # noqa: E402
from locma.policies.random_policy import RandomPolicy  # noqa: E402


def test_env_reset_step():
    env = BattleEnv(opponent=RandomPolicy("opp"), seed=0)
    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    mask = env.action_masks()
    assert mask.any()
    idx = int(np.argmax(mask))
    obs, reward, terminated, truncated, info = env.step(idx)
    assert reward in (-1.0, 0.0, 1.0)


def test_obs_size_matches_encode():
    """Verify OBS_SIZE constant matches encode_battle output length."""
    from locma.core.views import BattleView, CardView  # noqa: PLC0415

    dummy_card = CardView(
        instance_id=1, card_id=1, type=0, cost=1, attack=2, defense=3, abilities="------"
    )
    view = BattleView(
        turn=1,
        me_health=30,
        me_mana=1,
        op_health=30,
        op_hand_count=5,
        my_hand=(dummy_card,),
        my_board=(dummy_card,),
        op_board=(dummy_card,),
    )
    encoded = encode_battle(view)
    assert len(encoded) == OBS_SIZE, f"encode_battle returned {len(encoded)}, expected {OBS_SIZE}"
    assert encoded.dtype == np.float32
