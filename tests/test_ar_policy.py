import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

from locma.envs.action_factor import ACTION_SIZE  # noqa: E402
from locma.envs.ar_policy import MaskableAutoregressivePolicy  # noqa: E402


def _policy():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(308,), dtype=np.float32)
    act_space = spaces.Discrete(ACTION_SIZE)
    return MaskableAutoregressivePolicy(obs_space, act_space, lambda _: 3e-4)


def _masks(b):
    m = np.zeros((b, ACTION_SIZE), dtype=bool)
    m[:, 0] = True
    m[:, 1] = True
    m[:, 113] = True
    return m


def test_forward_returns_legal_actions_and_finite():
    torch.manual_seed(0)
    policy = _policy()
    obs = torch.randn(4, 308)
    masks = _masks(4)
    actions, values, log_prob = policy.forward(obs, action_masks=masks)
    assert actions.shape == (4,)
    for i, a in enumerate(actions.tolist()):
        assert masks[i, a]
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_prob).all()


def test_evaluate_actions_grads_flow_to_heads():
    torch.manual_seed(1)
    policy = _policy()
    obs = torch.randn(4, 308)
    masks = _masks(4)
    actions, _, _ = policy.forward(obs, action_masks=masks)
    values, log_prob, entropy = policy.evaluate_actions(obs, actions, action_masks=masks)
    loss = -(log_prob.mean()) + values.mean() - entropy.mean()
    loss.backward()
    assert policy.ar_heads.head_type.weight.grad is not None
    assert torch.isfinite(entropy).all()


def test_predict_is_deterministic_and_legal():
    torch.manual_seed(2)
    policy = _policy()
    obs = torch.randn(3, 308)
    masks = _masks(3)
    a1 = policy._predict(obs, deterministic=True, action_masks=masks)
    a2 = policy._predict(obs, deterministic=True, action_masks=masks)
    assert torch.equal(a1, a2)
    for i, a in enumerate(a1.tolist()):
        assert masks[i, a]
