"""MaskablePPO training entrypoint (requires the [ml] extra).

Drives the `locma train` CLI command. The training loop lives here so it has a
single home. Supports parallel envs (CPU speedup), a seeded trainer
(reproducibility), and intermediate checkpoints saved along one trajectory.
"""

from __future__ import annotations

import functools


def _make_battle_env(
    opponent_spec: str,
    seed: int,
    agent_seat: int = 0,
    seat_random: bool = False,
    obs_mode: str = "flat",
    reward_mode: str = "sparse",
):
    """Top-level env factory (picklable for SubprocVecEnv spawn on Windows).

    Rebuilds the opponent from its spec string inside each subprocess, so the
    (possibly stateful) opponent never has to be pickled and crosses no process
    boundary. ``seat_random`` trains the agent as both first and second player.
    ``obs_mode`` selects the observation encoding: "flat", "token", or
    "tactical"; "base" is accepted by BattleEnv as an alias for "flat".
    """
    from locma.envs.battle_env import BattleEnv  # noqa: PLC0415 — optional [ml] dep
    from locma.policies.registry import make_policy  # noqa: PLC0415

    return BattleEnv(
        opponent=make_policy(opponent_spec),
        seed=seed,
        agent_seat=agent_seat,
        seat_random=seat_random,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
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
):
    """Build a (vectorised) training env. n_envs>1 runs each env in its own
    process for true CPU parallelism; each env gets a distinct seed. ``both_seat``
    randomizes the agent's seat per episode (the +0.06-and-2x-efficiency fix).
    ``obs_mode`` selects the observation encoding."""
    from stable_baselines3.common.vec_env import (  # noqa: PLC0415
        DummyVecEnv,
        SubprocVecEnv,
    )

    fns = [
        functools.partial(
            _make_battle_env, opponent_spec, seed + i, 0, both_seat, obs_mode, reward_mode
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
    """Construct or load a MaskablePPO model, selecting policy class by obs_mode."""
    from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

    if init_model:
        model = MaskablePPO.load(init_model, env=env, seed=seed)
        model.verbose = verbose
        model.ent_coef = ent_coef
        model.learning_rate = learning_rate
        model.target_kl = target_kl
        return model

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
    both_seat: bool = True,
    obs_mode: str = "flat",
    reward_mode: str = "sparse",
    learning_rate: float = 3e-4,
    target_kl: float | None = None,
    init_model: str | None = None,
):
    """Train a seeded MaskablePPO agent against `opponent_spec` and save it.

    Imports the ML stack lazily; an ImportError means the `[ml]` extra is absent.
    """
    env = _build_env(
        opponent_spec, seed, n_envs, both_seat=both_seat, obs_mode=obs_mode, reward_mode=reward_mode
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


# A small, code-declared "zoo" of opponents to train against back-to-back. This is
# intentionally a constant for now (no CLI list plumbing); edit it to change the
# curriculum. Order matters — training proceeds left to right. See docs/cli.md and
# the future-explorations roadmap in docs/ppo-review.md.
ZOO_OPPONENTS: tuple[str, ...] = ("greedy", "scripted", "max-guard", "max-attack")


def train_zoo(
    opponents=ZOO_OPPONENTS,
    steps_per_opponent: int = 200_000,
    out: str = "model.zip",
    seed: int = 0,
    ent_coef: float = 0.02,
    verbose: int = 1,
    both_seat: bool = True,
    obs_mode: str = "flat",
    reward_mode: str = "sparse",
    learning_rate: float = 3e-4,
    target_kl: float | None = None,
    init_model: str | None = None,
):
    """Train ONE MaskablePPO model back-to-back against each opponent in turn."""
    opps = list(opponents)
    if not opps:
        raise ValueError("train_zoo needs a non-empty opponent list")

    model = _make_model(
        _build_env(opps[0], seed, 1, both_seat=both_seat, obs_mode=obs_mode, reward_mode=reward_mode),
        obs_mode=obs_mode,
        seed=seed,
        verbose=verbose,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
        init_model=init_model,
    )
    for i, opp in enumerate(opps):
        if i > 0:
            model.set_env(
                _build_env(
                    opp,
                    seed,
                    1,
                    both_seat=both_seat,
                    obs_mode=obs_mode,
                    reward_mode=reward_mode,
                )
            )
        model.learn(total_timesteps=steps_per_opponent, reset_num_timesteps=(i == 0))
    model.save(out)
    return out
