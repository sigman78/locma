"""MaskablePPO training entrypoint (requires the [ml] extra).

Drives the `locma train` CLI command. The training loop lives here so it has a
single home. Supports parallel envs (CPU speedup), a seeded trainer
(reproducibility), and intermediate checkpoints saved along one trajectory.
"""

from __future__ import annotations

import functools


def _normalize_obs_mode(obs_mode: str) -> str:
    return "flat" if obs_mode == "base" else obs_mode


def _make_battle_env(
    opponent_spec: str,
    seed: int,
    agent_seat: int = 0,
    seat_random: bool = False,
    obs_mode: str = "flat",
    reward_mode: str = "sparse",
    reward_scale: float = 0.05,
):
    """Top-level env factory (picklable for SubprocVecEnv spawn on Windows)."""
    from locma.envs.battle_env import BattleEnv  # noqa: PLC0415 — optional [ml] dep
    from locma.policies.registry import make_policy  # noqa: PLC0415

    return BattleEnv(
        opponent=make_policy(opponent_spec),
        seed=seed,
        agent_seat=agent_seat,
        seat_random=seat_random,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        reward_scale=reward_scale,
    )


def _ckpt_path(out: str, steps: int) -> str:
    """Derive a step-suffixed checkpoint path: model.zip + 1000 -> model-1000.zip."""
    base = out[:-4] if out.endswith(".zip") else out
    return f"{base}-{steps}.zip"


def _build_env(
    opponent_spec: str,
    seed: int,
    n_envs: int,
    both_seat: bool = True,
    obs_mode: str = "flat",
    reward_mode: str = "sparse",
    reward_scale: float = 0.05,
):
    """Build a vectorized training env."""
    from stable_baselines3.common.vec_env import (  # noqa: PLC0415
        DummyVecEnv,
        SubprocVecEnv,
    )

    fns = [
        functools.partial(
            _make_battle_env,
            opponent_spec,
            seed + i,
            0,
            both_seat,
            obs_mode,
            reward_mode,
            reward_scale,
        )
        for i in range(n_envs)
    ]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


def _make_model(
    env,
    *,
    obs_mode: str,
    seed: int,
    verbose: int,
    ent_coef: float,
    learning_rate: float = 3e-4,
    target_kl: float | None = None,
    init_model: str | None = None,
):
    """Construct or load a MaskablePPO model."""
    from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

    if init_model:
        model = MaskablePPO.load(init_model, env=env, seed=seed)
        model.verbose = verbose
        model.ent_coef = ent_coef
        _set_learning_rate(model, learning_rate)
        model.target_kl = target_kl
        return model

    obs_mode = _normalize_obs_mode(obs_mode)
    if obs_mode == "token":
        from locma.envs.extractor import TokenSetExtractor  # noqa: PLC0415

        return MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=dict(features_extractor_class=TokenSetExtractor),
            verbose=verbose,
            seed=seed,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            target_kl=target_kl,
        )

    return MaskablePPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        seed=seed,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
    )


def train_agent(
    opponent_spec: str,
    steps: int = 50_000,
    out: str = "model.zip",
    seed: int = 0,
    verbose: int = 1,
    n_envs: int = 1,
    checkpoints=None,
    ent_coef: float = 0.02,
    learning_rate: float = 3e-4,
    both_seat: bool = True,
    obs_mode: str = "flat",
    reward_mode: str = "sparse",
    reward_scale: float = 0.05,
    target_kl: float | None = None,
    init_model: str | None = None,
):
    """Train a seeded MaskablePPO agent against `opponent_spec` and save it."""
    env = _build_env(
        opponent_spec,
        seed,
        n_envs,
        both_seat=both_seat,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        reward_scale=reward_scale,
    )
    model = _make_model(
        env,
        obs_mode=obs_mode,
        seed=seed,
        verbose=verbose,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
        init_model=init_model,
    )

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


ZOO_OPPONENTS: tuple[str, ...] = ("greedy", "scripted", "max-guard", "max-attack")


def train_zoo(
    opponents=ZOO_OPPONENTS,
    steps_per_opponent: int = 200_000,
    out: str = "model.zip",
    seed: int = 0,
    ent_coef: float = 0.02,
    learning_rate: float = 3e-4,
    verbose: int = 1,
    n_envs: int = 1,
    both_seat: bool = True,
    obs_mode: str = "flat",
    reward_mode: str = "sparse",
    reward_scale: float = 0.05,
    target_kl: float | None = None,
    init_model: str | None = None,
):
    """Train one MaskablePPO model back-to-back against each opponent in turn."""
    return train_schedule(
        opponents=list(opponents),
        steps_per_phase=steps_per_opponent,
        out=out,
        seed=seed,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        verbose=verbose,
        n_envs=n_envs,
        both_seat=both_seat,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        reward_scale=reward_scale,
        target_kl=target_kl,
        init_model=init_model,
    )


def train_schedule(
    opponents: list[str],
    steps_per_phase: int = 100_000,
    out: str = "model.zip",
    seed: int = 0,
    ent_coef: float = 0.02,
    learning_rate: float = 3e-4,
    verbose: int = 1,
    n_envs: int = 1,
    both_seat: bool = True,
    obs_mode: str = "flat",
    reward_mode: str = "sparse",
    reward_scale: float = 0.05,
    target_kl: float | None = None,
    init_model: str | None = None,
):
    """Train one PPO model through an explicit opponent schedule."""
    if not opponents:
        raise ValueError("train_schedule needs a non-empty opponent list")

    model = _make_model(
        _build_env(
            opponents[0],
            seed,
            n_envs,
            both_seat=both_seat,
            obs_mode=obs_mode,
            reward_mode=reward_mode,
            reward_scale=reward_scale,
        ),
        obs_mode=obs_mode,
        seed=seed,
        verbose=verbose,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
        init_model=init_model,
    )
    for i, opp in enumerate(opponents):
        if i > 0:
            model.set_env(
                _build_env(
                    opp,
                    seed,
                    n_envs,
                    both_seat=both_seat,
                    obs_mode=obs_mode,
                    reward_mode=reward_mode,
                    reward_scale=reward_scale,
                )
            )
        model.learn(total_timesteps=steps_per_phase, reset_num_timesteps=(i == 0))
    model.save(out)
    return out


def _set_learning_rate(model, learning_rate: float) -> None:
    from stable_baselines3.common.utils import get_schedule_fn  # noqa: PLC0415

    model.learning_rate = learning_rate
    model.lr_schedule = get_schedule_fn(learning_rate)
