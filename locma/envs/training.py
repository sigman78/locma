"""MaskablePPO training entrypoint (requires the [ml] extra).

Shared by the `locma train` CLI command and the standalone `train.py` script so
the training loop lives in exactly one place.
"""

from __future__ import annotations


def train_agent(
    opponent,
    steps: int = 50_000,
    out: str = "model.zip",
    seed: int = 0,
    verbose: int = 1,
) -> str:
    """Train a MaskablePPO agent against `opponent` and save it to `out`.

    Returns the saved model path. Imports the ML stack lazily so that merely
    importing this module never pulls in gymnasium/sb3/torch; an ImportError
    here means the `[ml]` extra is not installed.
    """
    from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

    from locma.envs.battle_env import BattleEnv  # noqa: PLC0415 — optional [ml] dep

    env = BattleEnv(opponent=opponent, seed=seed)
    model = MaskablePPO("MlpPolicy", env, verbose=verbose)
    model.learn(total_timesteps=steps)
    model.save(out)
    env.close()
    return out
