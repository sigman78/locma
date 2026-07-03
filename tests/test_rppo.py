"""Tests for the maskable+recurrent PPO hybrid (locma/envs/rppo.py).

The training-path tests are gated with importorskip (skipped without the [ml]
extra) and marked slow (opt-in via `pytest -m slow`), like the other
model-training tests. The registry test is fast and dependency-free.
"""

from __future__ import annotations

import numpy as np
import pytest

from locma.policies.registry import is_policy_spec, make_policy


def test_rppo_spec_registered():
    """`rppo:` is a registered spec and builds lazily (no [ml] import, no file)."""
    assert is_policy_spec("rppo:runs/nonexistent.zip")
    policy = make_policy("rppo:runs/nonexistent.zip")
    assert policy.name == "rppo:runs/nonexistent.zip"
    assert policy.draft is not None  # balanced draft pairing, like ppo:


pytest.importorskip("sb3_contrib")
pytest.importorskip("torch")

# Long model-training/game-playing tests below: opt-in via `pytest -m slow`.


@pytest.mark.slow
def test_train_recurrent_flat_smoke(tmp_path):
    """Flat path: recurrent=True trains, saves, reloads, and respects masks."""
    from locma.envs.rppo import MaskableRecurrentPPO  # noqa: PLC0415 -- [ml]-gated
    from locma.envs.training import train_agent  # noqa: PLC0415 -- [ml]-gated

    out = str(tmp_path / "rppo-flat.zip")
    train_agent(
        "random",
        steps=256,
        out=out,
        seed=0,
        n_envs=1,
        verbose=0,
        n_steps=128,
        batch_size=32,
        recurrent=True,
        lstm_kwargs={"lstm_hidden_size": 32},
    )
    model = MaskableRecurrentPPO.load(out)
    assert model.policy.lstm_actor.hidden_size == 32

    # Masking respected: only action 0 (Pass) legal -> predict must return 0,
    # regardless of hidden state; state must round-trip through predict.
    obs = np.zeros(model.observation_space.shape, dtype=np.float32)
    mask = np.zeros(model.action_space.n, dtype=bool)
    mask[0] = True
    state = None
    for episode_start in (True, False):
        idx, state = model.predict(
            obs,
            state=state,
            episode_start=np.array([episode_start]),
            deterministic=True,
            action_masks=mask,
        )
        assert int(idx) == 0
        assert state is not None and len(state) == 2


@pytest.mark.slow
def test_train_recurrent_token_and_play(tmp_path):
    """Token path: recurrent training saves a Dict-obs model, and the rppo:
    battle policy plays full games through the harness (stateful predict)."""
    import gymnasium  # noqa: PLC0415 -- [ml]-gated

    from locma.envs.rppo import MaskableRecurrentPPO  # noqa: PLC0415 -- [ml]-gated
    from locma.envs.training import train_agent  # noqa: PLC0415 -- [ml]-gated
    from locma.harness.match import run_match  # noqa: PLC0415 -- [ml]-gated

    out = str(tmp_path / "rppo-tok.zip")
    train_agent(
        "random",
        steps=256,
        out=out,
        seed=0,
        n_envs=1,
        obs_mode="token",
        verbose=0,
        n_steps=128,
        batch_size=32,
        recurrent=True,
        lstm_kwargs={"lstm_hidden_size": 32},
    )
    model = MaskableRecurrentPPO.load(out)
    assert isinstance(model.observation_space, gymnasium.spaces.Dict)

    result = run_match(make_policy(f"rppo:{out}"), make_policy("random"), games=2, seed=0)
    assert result.games == 4  # mirrored
