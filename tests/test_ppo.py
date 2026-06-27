import types

import numpy as np
import pytest

from locma.policies.ppo import MaskablePPOBattlePolicy


def test_constructs_without_loading_model():
    # Construction must not touch the file or import sb3 (lazy).
    p = MaskablePPOBattlePolicy(model_path="nonexistent.zip", name="ppo")
    assert p.name == "ppo"
    assert p.model_path == "nonexistent.zip"
    assert p._model is None


def test_battle_action_signature_accepts_state():
    p = MaskablePPOBattlePolicy(model_path="nonexistent.zip")
    # We don't call it (no model); just assert the attribute contract.
    assert p.deterministic is True
    assert hasattr(p, "battle_action")
    assert not hasattr(p, "draft_action")  # pure battle policy now


def test_encode_for_selects_encoder_by_obs_space():
    """_encode_for routes to encode_battle_tokens for Dict obs, encode_battle for Box."""
    pytest.importorskip("gymnasium")

    from gymnasium import spaces  # noqa: PLC0415

    from locma.core.views import BattleView, CardView  # noqa: PLC0415
    from locma.envs.encode import token_obs_space  # noqa: PLC0415
    from locma.policies.ppo import _encode_for  # noqa: PLC0415

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

    dict_stub = types.SimpleNamespace(observation_space=token_obs_space())
    box_stub = types.SimpleNamespace(
        observation_space=spaces.Box(low=-1, high=1, shape=(308,), dtype=np.float32)
    )

    result_dict = _encode_for(dict_stub, view)
    assert isinstance(result_dict, dict), "Dict obs space should return a dict"
    assert "tokens" in result_dict
    assert "card_ids" in result_dict

    result_box = _encode_for(box_stub, view)
    assert isinstance(result_box, np.ndarray), "Box obs space should return an ndarray"
    assert result_box.shape == (308,)
