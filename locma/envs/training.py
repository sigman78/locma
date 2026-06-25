"""MaskablePPO training entrypoint (requires the [ml] extra).

Drives the `locma train` CLI command. The training loop lives here so it has a
single home. Supports parallel envs (CPU speedup), a seeded trainer
(reproducibility), and intermediate checkpoints saved along one trajectory.
"""

from __future__ import annotations

import functools


def _make_battle_env(opponent_spec: str, seed: int, agent_seat: int = 0):
    """Top-level env factory (picklable for SubprocVecEnv spawn on Windows).

    Rebuilds the opponent from its spec string inside each subprocess, so the
    (possibly stateful) opponent never has to be pickled and crosses no process
    boundary.
    """
    from locma.envs.battle_env import BattleEnv  # noqa: PLC0415 — optional [ml] dep
    from locma.policies.registry import make_policy  # noqa: PLC0415

    return BattleEnv(opponent=make_policy(opponent_spec), seed=seed, agent_seat=agent_seat)


def _ckpt_path(out: str, steps: int) -> str:
    """Derive a step-suffixed checkpoint path: model.zip + 1000 -> model-1000.zip."""
    base = out[:-4] if out.endswith(".zip") else out
    return f"{base}-{steps}.zip"


def _build_env(opponent_spec: str, seed: int, n_envs: int):
    """Build a (vectorised) training env. n_envs>1 runs each env in its own
    process for true CPU parallelism; each env gets a distinct seed."""
    from stable_baselines3.common.vec_env import (  # noqa: PLC0415
        DummyVecEnv,
        SubprocVecEnv,
    )

    fns = [functools.partial(_make_battle_env, opponent_spec, seed + i) for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


def train_agent(
    opponent_spec: str,
    steps: int = 50_000,
    out: str = "model.zip",
    seed: int = 0,
    verbose: int = 1,
    n_envs: int = 1,
    checkpoints=None,
    ent_coef: float = 0.02,
):
    """Train a seeded MaskablePPO agent against `opponent_spec` and save it.

    Parameters
    ----------
    opponent_spec: registry spec string for the opponent (rebuilt per env).
    steps: total env timesteps (ignored when `checkpoints` is given).
    out: output model path; checkpoints derive step-suffixed siblings.
    n_envs: number of parallel envs (CPU speedup).
    checkpoints: optional iterable of step marks. When given, training runs as
        one continuous trajectory, saving a step-suffixed model at each mark, and
        returns the list of saved paths. Otherwise trains `steps` and returns
        the single `out` path.

    Imports the ML stack lazily; an ImportError means the `[ml]` extra is absent.
    """
    from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

    env = _build_env(opponent_spec, seed, n_envs)
    model = MaskablePPO("MlpPolicy", env, verbose=verbose, seed=seed, ent_coef=ent_coef)

    if checkpoints:
        marks = sorted({int(m) for m in checkpoints})
        prev = 0
        saved = []
        for i, mark in enumerate(marks):
            model.learn(total_timesteps=mark - prev, reset_num_timesteps=(i == 0))
            path = _ckpt_path(out, mark)
            model.save(path)
            saved.append(path)
            prev = mark
        env.close()
        return saved

    model.learn(total_timesteps=steps)
    model.save(out)
    env.close()
    return out
