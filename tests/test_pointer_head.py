"""Tests for locma.envs.pointer_head — torch/SB3 importorskip-gated.

Uses the REAL BattleEnv token obs space (the pointer head's action table is
coupled to the 155-action / 20-slot layout, so a toy env would test nothing),
but only tiny learn() budgets — seconds, not minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from sb3_contrib import MaskablePPO  # noqa: E402

from locma.envs.encode import ACTION_SIZE, MAX_TOKENS  # noqa: E402
from locma.envs.pointer_head import (  # noqa: E402
    NONE_SLOT,
    PointerMaskablePolicy,
    build_action_table,
)
from locma.envs.training import _build_env, _make_model  # noqa: E402


def test_action_table_layout():
    src, tgt, fam = build_action_table()
    assert src.shape == tgt.shape == fam.shape == (ACTION_SIZE,)
    assert fam[0] == 0 and src[0] == NONE_SLOT  # pass
    assert list(src[1:9]) == list(range(8)) and set(fam[1:9]) == {1}  # summon
    # use hand[2] -> op_board[3]: idx 9 + 2*13 + 9
    i = 9 + 2 * 13 + 9
    assert (src[i], tgt[i], fam[i]) == (2, 14 + 3, 2)
    # use hand[5] -> face/none
    i = 9 + 5 * 13 + 12
    assert (src[i], tgt[i], fam[i]) == (5, NONE_SLOT, 2)
    # attack my_board[4] -> face: idx 113 + 4*7 + 6
    i = 113 + 4 * 7 + 6
    assert (src[i], tgt[i], fam[i]) == (8 + 4, NONE_SLOT, 3)
    assert src.max() < MAX_TOKENS + 1


@pytest.fixture(scope="module")
def pointer_model():
    env = _build_env("random", seed=0, n_envs=1, both_seat=False, obs_mode="token")
    model = _make_model(env, obs_mode="token", seed=0, verbose=0, ent_coef=0.0, pointer_head=True)
    yield model
    env.close()


def test_build_and_predict(pointer_model):
    assert isinstance(pointer_model.policy, PointerMaskablePolicy)
    obs = pointer_model.env.reset()
    action, _ = pointer_model.predict(obs, deterministic=True)
    assert 0 <= int(action[0]) < ACTION_SIZE


def test_learn_smoke(pointer_model):
    pointer_model.learn(total_timesteps=64)  # one tiny rollout + update


def test_save_load_roundtrip(tmp_path, pointer_model):
    p = str(tmp_path / "pointer.zip")
    pointer_model.save(p)
    # Same device as the original: near-uniform logits at this training budget
    # make argmax sensitive to cpu-vs-cuda float noise, which is not what this
    # test is about (state round-trip is).
    loaded = MaskablePPO.load(p, device=pointer_model.device)
    assert isinstance(loaded.policy, PointerMaskablePolicy)
    # eval mode both: the extractor's dropout is active in train mode and
    # would make this comparison stochastic.
    pointer_model.policy.set_training_mode(False)
    loaded.policy.set_training_mode(False)
    obs = pointer_model.env.reset()
    with torch.no_grad():
        obs_t1, _ = pointer_model.policy.obs_to_tensor(obs)
        a1, v1, _ = pointer_model.policy.forward(obs_t1, deterministic=True)
        obs_t2, _ = loaded.policy.obs_to_tensor(obs)
        a2, v2, _ = loaded.policy.forward(obs_t2, deterministic=True)
    assert int(a1[0]) == int(a2[0])
    assert np.allclose(v1.cpu().numpy(), v2.cpu().numpy(), atol=1e-5)


def test_pointer_requires_token():
    with pytest.raises(ValueError, match="token"):
        env = _build_env("random", seed=0, n_envs=1, both_seat=False, obs_mode="flat")
        try:
            _make_model(env, obs_mode="flat", seed=0, verbose=0, ent_coef=0.0, pointer_head=True)
        finally:
            env.close()
