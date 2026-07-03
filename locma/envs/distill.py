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

from locma.envs.encode import ACTION_SIZE, MAX_TOKENS, N_TACTICAL, OBS_SIZE, TOKEN_FEATS
from locma.envs.practicum import _manifest_path

_ARRAY_KEYS = ("obs", "action", "mask", "winner", "seat", "opponent_id", "game_id")
_TOKEN_ARRAY_KEYS = (
    "obs_tokens",
    "obs_card_ids",
    "obs_token_mask",
    "obs_scalars",
    "action",
    "mask",
    "winner",
    "seat",
    "opponent_id",
    "game_id",
)


def load_practicum(path: str) -> tuple[dict, dict]:
    """Load practicum arrays + manifest; reject a layout mismatch loudly.

    Returns (arrays, manifest).  For "flat" obs_mode the arrays dict contains
    ``obs``; for "token" obs_mode it contains the four token keys
    (``obs_tokens``, ``obs_card_ids``, ``obs_token_mask``, ``obs_scalars``).
    """
    with open(_manifest_path(path), encoding="utf-8") as f:
        manifest = json.load(f)

    obs_mode = manifest.get("obs_mode", "flat")

    if obs_mode == "token":
        mt = manifest.get("max_tokens")
        tf = manifest.get("token_feats")
        nt = manifest.get("n_tactical")
        az = manifest.get("action_size")
        if mt != MAX_TOKENS or tf != TOKEN_FEATS or nt != N_TACTICAL or az != ACTION_SIZE:
            raise ValueError(
                "practicum layout mismatch: manifest "
                f"max_tokens/token_feats/n_tactical/action_size="
                f"{mt}/{tf}/{nt}/{az} "
                f"!= encode {MAX_TOKENS}/{TOKEN_FEATS}/{N_TACTICAL}/{ACTION_SIZE}; "
                "regenerate the practicum"
            )
        with np.load(path) as data:
            arrays = {k: data[k] for k in _TOKEN_ARRAY_KEYS if k in data.files}
    else:
        # "flat" (default for back-compat with old manifests that lack obs_mode)
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
    ids = np.asarray(game_id).tolist()
    uniq = sorted({int(g) for g in ids})
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac) if len(uniq) > 1 else 0
    val_games = set(uniq[:n_val])
    train_idx = [i for i, g in enumerate(ids) if int(g) not in val_games]
    val_idx = [i for i, g in enumerate(ids) if int(g) in val_games]
    return train_idx, val_idx


def _load_init_model(init_model: str, want_dict_obs: bool):
    """Load a warm-start model and reject an obs-family mismatch loudly."""
    import gymnasium  # noqa: PLC0415 — optional [ml] dep
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415

    model = MaskablePPO.load(resolve_path(init_model))
    is_dict = isinstance(model.observation_space, gymnasium.spaces.Dict)
    if is_dict != want_dict_obs:
        want = "token (Dict)" if want_dict_obs else "flat (Box)"
        raise ValueError(
            f"init_model {init_model!r} obs space does not match practicum: need {want}"
        )
    return model


def behavior_clone(
    data: str = "practicum.npz",
    out: str = "model.zip",
    epochs: int = 10,
    batch: int = 256,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    seed: int = 0,
    verbose: int = 1,
    obs_mode: str | None = None,
    init_model: str | None = None,
) -> dict:
    """Masked-CE behavior cloning of a practicum into a MaskablePPO model.zip.

    ``obs_mode`` defaults to the value recorded in the practicum manifest.  If
    supplied it must match the manifest — mismatches raise a clear ``ValueError``
    rather than a cryptic ``KeyError``.
    ``obs_mode="flat"`` uses MlpPolicy and a flat obs vector.
    ``obs_mode="token"`` uses MultiInputPolicy + TokenSetExtractor and a dict obs.
    ``init_model`` warm-starts from a saved model.zip (or ``depot:`` ref) of the
    matching obs mode instead of a fresh net — ALL parameters train (the critic
    drifts with the shared extractor; use ``vbeam_distill.train_policy_head``
    when the critic must stay intact).
    """
    arrays, manifest = load_practicum(data)

    # Resolve obs_mode from manifest — before any ML import so mismatches are cheap.
    manifest_mode = manifest.get("obs_mode", "flat")
    if obs_mode is None:
        obs_mode = manifest_mode
    elif obs_mode != manifest_mode:
        raise ValueError(
            f"obs_mode={obs_mode!r} but practicum manifest is {manifest_mode!r}; "
            f"omit --obs-mode or pass --obs-mode {manifest_mode!r}"
        )

    import torch  # noqa: PLC0415 — optional [ml] dep
    from sb3_contrib import MaskablePPO  # noqa: PLC0415
    from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: PLC0415

    from locma.envs.training import _make_battle_env  # noqa: PLC0415

    act = arrays["action"].astype(np.int64)
    mask = arrays["mask"].astype(bool)

    train_idx, val_idx = split_by_game(arrays["game_id"], val_frac, seed)

    if obs_mode == "token":
        if init_model is not None:
            model = _load_init_model(init_model, want_dict_obs=True)
        else:
            from locma.envs.extractor import TokenSetExtractor  # noqa: PLC0415

            # Throwaway env to supply the Dict obs/action spaces + policy architecture.
            env = DummyVecEnv([lambda: _make_battle_env("random", seed, obs_mode="token")])
            model = MaskablePPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=dict(features_extractor_class=TokenSetExtractor),
                seed=seed,
                verbose=0,
            )
            env.close()

        device = model.device
        # Build per-key tensors mapping practicum storage names → obs-space keys.
        obs_t = {
            "tokens": torch.as_tensor(arrays["obs_tokens"].astype(np.float32), device=device),
            "card_ids": torch.as_tensor(arrays["obs_card_ids"].astype(np.float32), device=device),
            "token_mask": torch.as_tensor(
                arrays["obs_token_mask"].astype(np.float32), device=device
            ),
            "scalars": torch.as_tensor(arrays["obs_scalars"].astype(np.float32), device=device),
        }
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
                batch_obs = {k: v[sel] for k, v in obs_t.items()}
                _, log_prob, _ = model.policy.evaluate_actions(
                    batch_obs, act_t[sel], action_masks=mask_t[sel]
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
            pred, _ = model.predict(
                {
                    "tokens": arrays["obs_tokens"][v],
                    "card_ids": arrays["obs_card_ids"][v],
                    "token_mask": arrays["obs_token_mask"][v],
                    "scalars": arrays["obs_scalars"][v],
                },
                action_masks=mask[v],
                deterministic=True,
            )
            val_agreement = float((np.asarray(pred).reshape(-1) == act[v]).mean())

    else:
        # "flat" path — unchanged from the original implementation.
        obs = arrays["obs"].astype(np.float32)

        if init_model is not None:
            model = _load_init_model(init_model, want_dict_obs=False)
        else:
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
