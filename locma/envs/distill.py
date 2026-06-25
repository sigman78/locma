"""Distillation: behavior-clone a practicum into a MaskablePPO model.zip.

Pure helpers (dataset loading with a layout guard, game-level split) are unit
tested. The masked cross-entropy loop trains the real MaskablePPO policy net via
sb3-contrib's masked distribution and is verified operationally (runbook), in line
with how training.py's learn loop is treated.
"""

from __future__ import annotations

import json
import random

import numpy as np

from locma.envs.encode import ACTION_SIZE, OBS_SIZE
from locma.envs.practicum import _manifest_path

_ARRAY_KEYS = ("obs", "action", "mask", "winner", "seat", "opponent_id", "game_id")


def load_practicum(path: str) -> tuple[dict, dict]:
    """Load practicum arrays + manifest; reject a layout mismatch loudly."""
    with open(_manifest_path(path), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("obs_size") != OBS_SIZE or manifest.get("action_size") != ACTION_SIZE:
        raise ValueError(
            "practicum layout mismatch: manifest "
            f"obs/action={manifest.get('obs_size')}/{manifest.get('action_size')} "
            f"!= encode {OBS_SIZE}/{ACTION_SIZE}; regenerate the practicum"
        )
    with np.load(path) as data:
        arrays = {k: data[k] for k in _ARRAY_KEYS if k in data.files}
    return arrays, manifest


def split_by_game(game_id, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Split row indices so no game appears in both train and val."""
    game_id = np.asarray(game_id)
    uniq = sorted({int(g) for g in game_id.tolist()})
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac) if len(uniq) > 1 else 0
    val_games = set(uniq[:n_val])
    train_idx = [i for i, g in enumerate(game_id.tolist()) if int(g) not in val_games]
    val_idx = [i for i, g in enumerate(game_id.tolist()) if int(g) in val_games]
    return train_idx, val_idx


def behavior_clone(
    data: str = "practicum.npz",
    out: str = "model.zip",
    epochs: int = 10,
    batch: int = 256,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    seed: int = 0,
    verbose: int = 1,
) -> dict:
    """Masked-CE behavior cloning of a practicum into a MaskablePPO model.zip."""
    import torch  # noqa: PLC0415 — optional [ml] dep
    from sb3_contrib import MaskablePPO  # noqa: PLC0415
    from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: PLC0415

    from locma.envs.training import _make_battle_env  # noqa: PLC0415

    arrays, _ = load_practicum(data)
    obs = arrays["obs"].astype(np.float32)
    act = arrays["action"].astype(np.int64)
    mask = arrays["mask"].astype(bool)

    train_idx, val_idx = split_by_game(arrays["game_id"], val_frac, seed)

    # A throwaway env only supplies the obs/action spaces + policy architecture.
    env = DummyVecEnv([lambda: _make_battle_env("random", seed)])
    model = MaskablePPO("MlpPolicy", env, seed=seed, verbose=0)
    env.close()

    device = model.device
    obs_t = torch.as_tensor(obs, device=device)
    act_t = torch.as_tensor(act, device=device)
    mask_t = torch.as_tensor(mask, device=device)

    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    model.policy.set_training_mode(True)
    rng = np.random.default_rng(seed)
    order = np.asarray(train_idx, dtype=np.int64)

    for epoch in range(epochs):
        rng.shuffle(order)
        total = 0.0
        nb = 0
        for start in range(0, len(order), batch):
            sel = torch.as_tensor(order[start : start + batch], device=device)
            _, log_prob, _ = model.policy.evaluate_actions(
                obs_t[sel], act_t[sel], action_masks=mask_t[sel]
            )
            loss = -log_prob.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  loss={total / max(nb, 1):.4f}")

    # Top-1 agreement on held-out games.
    val_agreement = float("nan")
    if val_idx:
        v = np.asarray(val_idx, dtype=np.int64)
        pred, _ = model.predict(obs[v], action_masks=mask[v], deterministic=True)
        val_agreement = float((np.asarray(pred).reshape(-1) == act[v]).mean())

    model.save(out)
    return {
        "out": out,
        "val_agreement": val_agreement,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "epochs": epochs,
    }
