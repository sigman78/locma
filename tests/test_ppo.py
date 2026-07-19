import random
import types

import numpy as np
import pytest

from locma.policies.ppo import MaskablePPOBattlePolicy, MaskablePPOEnsembleBattlePolicy


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
    assert "token_mask" in result_dict
    assert "scalars" in result_dict

    result_box = _encode_for(box_stub, view)
    assert isinstance(result_box, np.ndarray), "Box obs space should return an ndarray"
    assert result_box.shape == (308,)


def test_token_model_save_load_produces_legal_action(tmp_path):
    """Close the env→save→load→eval loop for a token obs model without training.

    Build a token env + MaskablePPO model, save without learning, load via
    MaskablePPOBattlePolicy, and confirm the policy returns a legal action.
    This exercises the Dict-obs predict(..., action_masks=...) path end-to-end.
    """
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")

    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.engine import make_battle_view  # noqa: PLC0415
    from locma.envs.battle_env import BattleEnv  # noqa: PLC0415
    from locma.envs.training import _build_env, _make_model  # noqa: PLC0415
    from locma.policies.battles import RandomBattlePolicy  # noqa: PLC0415
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.drafts import RandomDraftPolicy  # noqa: PLC0415

    # Build a token env and model, save without learning.
    env = _build_env("random", seed=0, n_envs=1, obs_mode="token")
    model = _make_model(env, obs_mode="token", seed=0, verbose=0, ent_coef=0.02)
    path = str(tmp_path / "tok.zip")
    model.save(path)
    env.close()

    # Build a real battle GameState by resetting a BattleEnv — the easiest way
    # to get a valid battle state without re-implementing the draft engine.
    opp = Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name="opp")
    battle_env = BattleEnv(opponent=opp, seed=42, obs_mode="token")
    battle_env.reset()
    gs = battle_env.gs

    view = make_battle_view(gs)
    legal = battlemod.battle_legal(gs)
    battle_env.close()

    # Load via MaskablePPOBattlePolicy and get a decision.
    pol = MaskablePPOBattlePolicy(model_path=path)
    action = pol.battle_action(view, legal, gs)
    assert action in legal, f"Policy returned illegal action {action!r}"


# ---------------------------------------------------------------------------
# MaskablePPOEnsembleBattlePolicy (E26)
# ---------------------------------------------------------------------------


def test_ensemble_constructs_without_loading_models():
    p = MaskablePPOEnsembleBattlePolicy(["a.zip", "b.zip"])
    assert p.model_paths == ["a.zip", "b.zip"]
    assert p._models is None


def test_ensemble_requires_two_members():
    with pytest.raises(ValueError, match="at least 2"):
        MaskablePPOEnsembleBattlePolicy(["a.zip"])


def test_ensemble_of_identical_models_matches_single_model_action(tmp_path):
    """An ensemble of copies of one model must reproduce that model's own
    masked argmax exactly (mean of identical distributions == the distribution)."""
    pytest.importorskip("sb3_contrib")
    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.draft import apply_draft_pick, start_draft  # noqa: PLC0415
    from locma.core.engine import make_battle_view  # noqa: PLC0415
    from locma.core.state import GameState, Phase  # noqa: PLC0415
    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.envs.training import _build_env, _make_model  # noqa: PLC0415

    env = _build_env("random", 0, 1, obs_mode="token")
    model = _make_model(env, obs_mode="token", seed=0, verbose=0, ent_coef=0.02)
    path = str(tmp_path / "m.zip")
    model.save(path)
    env.close()

    gs = GameState.new(random.Random(0))
    start_draft(gs, load_cards())
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
    battlemod.start_battle(gs)
    view = make_battle_view(gs)
    legal = battlemod.battle_legal(gs)

    single = MaskablePPOBattlePolicy(model_path=path)
    ens = MaskablePPOEnsembleBattlePolicy([path, path, path])

    a1 = single.battle_action(view, legal, gs)
    a2 = ens.battle_action(view, legal, gs)
    assert a1 == a2
    assert a2 in legal
