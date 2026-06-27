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
