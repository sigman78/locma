"""Smoke tests for the tokenized-obs training path (MultiInputPolicy branch).

Gated with importorskip so the test is skipped in CI (dev extra only); it runs
locally where the [ml] extra (torch + sb3_contrib) is present.

TDD notes
---------
RED  — written *before* obs_mode was threaded through training.py; test (a)
       fails with TypeError (unexpected keyword argument 'obs_mode'); test (b)
       passes because default obs_mode="flat" is the unchanged path.
GREEN — after training.py changes: both pass.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sb3_contrib")
pytest.importorskip("torch")
gymnasium = pytest.importorskip("gymnasium")

from sb3_contrib import MaskablePPO  # noqa: E402

from locma.envs.training import train_agent  # noqa: E402

# Long model-training/game-playing tests: opt-in via `pytest -m slow`.
pytestmark = pytest.mark.slow


def test_train_agent_token_smoke(tmp_path):
    """(a) Token path: train_agent with obs_mode='token' saves a Dict-obs model."""
    out = train_agent(
        "random",
        steps=512,
        out=str(tmp_path / "tok.zip"),
        seed=0,
        n_envs=1,
        obs_mode="token",
        verbose=0,
    )

    # File was saved.
    assert tmp_path.joinpath("tok.zip").exists(), f"model not found at {out}"

    # The saved model has a Dict observation space → confirms MultiInputPolicy path.
    m = MaskablePPO.load(out)
    assert isinstance(m.observation_space, gymnasium.spaces.Dict), (
        f"expected Dict obs space, got {type(m.observation_space)}"
    )


def test_train_agent_token_lr_and_target_kl(tmp_path):
    """(c) Token path: learning_rate and target_kl are accepted and applied."""
    out = str(tmp_path / "m.zip")
    train_agent(
        "random",
        steps=512,
        out=out,
        seed=0,
        n_envs=1,
        obs_mode="token",
        learning_rate=1e-4,
        target_kl=0.025,
        verbose=0,
    )

    assert tmp_path.joinpath("m.zip").exists(), "model not saved"

    # Confirm the model loads and has a Dict obs space (MultiInputPolicy path).
    m = MaskablePPO.load(out)
    assert isinstance(m.observation_space, gymnasium.spaces.Dict), (
        f"expected Dict obs space, got {type(m.observation_space)}"
    )
    # learning_rate is stored as a schedule callable; confirm it evaluates to 1e-4.
    lr = m.learning_rate
    actual_lr = lr(1.0) if callable(lr) else lr
    assert actual_lr == pytest.approx(1e-4), f"expected lr=1e-4, got {actual_lr}"


def test_train_agent_flat_still_works(tmp_path):
    """(b) Flat path unchanged after refactor: default obs_mode yields a Box-obs model."""
    out = train_agent(
        "random",
        steps=512,
        out=str(tmp_path / "flat.zip"),
        seed=0,
        n_envs=1,
        verbose=0,
    )

    assert tmp_path.joinpath("flat.zip").exists(), f"model not found at {out}"

    m = MaskablePPO.load(out)
    assert isinstance(m.observation_space, gymnasium.spaces.Box), (
        f"expected Box obs space, got {type(m.observation_space)}"
    )
