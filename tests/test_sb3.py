from __future__ import annotations

import pytest
pytest.importorskip("sb3_contrib")
from locma.policies.sb3_policy import SB3Policy


def test_sb3_policy_constructs_without_model():
    # name set; model loads lazily so construction must not require a file
    p = SB3Policy(model_path="nonexistent.zip", name="ppo")
    assert p.name == "ppo"
