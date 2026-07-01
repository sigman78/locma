"""Optuna hyperparameter sweep for the PPO ceiling study (requires [ml] + [sweep]).

The config layer (PPOConfig, sample_config, valid) is pure and import-safe; the
objective/driver (below) need the ML stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    target_kl: float | None = 0.025
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # token-extractor arch (Phase 1b)
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    features_dim: int = 256

    def to_train_kwargs(self) -> dict:
        d = asdict(self)
        arch = {k: d.pop(k) for k in ("d_model", "n_layers", "n_heads", "features_dim")}
        d["extractor_kwargs"] = arch
        return d


# The baseline point — enqueued into the study so TPE never re-derives it.
B0_CONFIG = PPOConfig()


def valid(cfg: PPOConfig, n_envs: int) -> bool:
    """SB3 requires the rollout buffer to divide into minibatches."""
    return cfg.batch_size <= cfg.n_steps * n_envs


def sample_config(trial, *, sweep_arch: bool = False) -> PPOConfig:
    cfg = PPOConfig(
        learning_rate=trial.suggest_float("learning_rate", 3e-5, 5e-4, log=True),
        # 0.025 (not 0.02) so B0_CONFIG's exact value is a member of this categorical
        # space -- run_sweep() enqueues B0 verbatim, and Optuna requires an enqueued
        # value to match a choice exactly or suggest_categorical raises ValueError.
        target_kl=trial.suggest_categorical("target_kl", [0.025, 0.03, 0.05, None]),
        n_steps=trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        n_epochs=trial.suggest_int("n_epochs", 3, 10),
        gamma=trial.suggest_categorical("gamma", [0.99, 0.995, 0.999]),
        gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        ent_coef=trial.suggest_float("ent_coef", 1e-3, 5e-2, log=True),
        vf_coef=trial.suggest_categorical("vf_coef", [0.5, 1.0]),
    )
    if sweep_arch:
        cfg.d_model = trial.suggest_categorical("d_model", [64, 128])
        cfg.n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
        cfg.n_heads = trial.suggest_categorical("n_heads", [4, 8])
        cfg.features_dim = trial.suggest_categorical("features_dim", [128, 256])
    return cfg


def objective(
    trial,
    *,
    n_envs: int,
    total_steps: int,
    eval_freq: int,
    n_games: int,
    sweep_arch: bool,
    tb_root: str,
    device: str,
) -> float:
    from locma.envs.eval_callback import WinRateEvalCallback  # noqa: PLC0415
    from locma.envs.training import train_zoo  # noqa: PLC0415

    cfg = sample_config(trial, sweep_arch=sweep_arch)
    if not valid(cfg, n_envs):
        return -1.0  # tell TPE this region is infeasible without crashing the worker

    steps_per_opp = max(1, total_steps // 4)  # 4-opponent zoo curriculum
    cb = WinRateEvalCallback(eval_freq=eval_freq, n_games=n_games, trial=trial)
    out = f"{tb_root}/trial_{trial.number}_model.zip"
    train_zoo(
        steps_per_opponent=steps_per_opp,
        out=out,
        seed=0,
        n_envs=n_envs,
        both_seat=True,
        obs_mode="token",
        device=device,
        tensorboard_log=f"{tb_root}/trial_{trial.number}",
        callback=cb,
        **cfg.to_train_kwargs(),
    )
    return float(cb.last_avg_hard3)


def run_sweep(
    *,
    storage: str,
    study_name: str,
    n_trials: int,
    n_envs: int = 8,
    total_steps: int = 300_000,
    eval_freq: int | None = None,
    n_games: int = 120,
    sweep_arch: bool = False,
    tb_root: str = "runs/tb",
    device: str = "auto",
):
    import optuna  # noqa: PLC0415
    from optuna.pruners import HyperbandPruner  # noqa: PLC0415
    from optuna.samplers import TPESampler  # noqa: PLC0415

    # eval_freq must be a multiple of the rollout size or the modulus check skips evals.
    if eval_freq is None:
        eval_freq = max(2048, (total_steps // 6) // (2048) * 2048) or 2048

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=0),
        pruner=HyperbandPruner(),
        load_if_exists=True,
    )
    # Seed TPE with the known-good baseline point (Phase 1a only — arch is fixed there).
    if not sweep_arch and not study.trials:
        b0 = asdict(B0_CONFIG)
        study.enqueue_trial(
            {k: b0[k] for k in (
                "learning_rate", "target_kl", "n_steps", "batch_size", "n_epochs",
                "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
            )}
        )

    study.optimize(
        lambda t: objective(
            t,
            n_envs=n_envs,
            total_steps=total_steps,
            eval_freq=eval_freq,
            n_games=n_games,
            sweep_arch=sweep_arch,
            tb_root=tb_root,
            device=device,
        ),
        n_trials=n_trials,
    )
    return study
