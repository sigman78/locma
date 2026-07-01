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
        target_kl=trial.suggest_categorical("target_kl", [0.02, 0.03, 0.05, None]),
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
