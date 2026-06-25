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
