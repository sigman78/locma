"""Warm-start AlphaZero trainer: load self-play data + fine-tune both heads.

``load_selfplay`` loads, layout-guards, and concatenates self-play ``.npz``
datasets produced by P3's ``record_selfplay``.  ``az_train`` warm-starts from
an existing token MaskablePPO net and trains both heads — policy via soft
cross-entropy to the recorded visit distribution, value via MSE to the
recorded game outcome.

``[ml]`` imports (torch / sb3) are lazy: this module is importable without
the ``[ml]`` extra.
"""

from __future__ import annotations

import json

import numpy as np

from locma.envs.encode import ACTION_SIZE, MAX_TOKENS, N_TACTICAL, TOKEN_FEATS
from locma.envs.practicum import _manifest_path

_SELFPLAY_ARRAY_KEYS = (
    "obs_tokens",
    "obs_card_ids",
    "obs_token_mask",
    "obs_scalars",
    "policy_target",
    "mask",
    "value_target",
    "seat",
    "game_id",
)


def load_selfplay(paths) -> tuple[dict, dict]:
    """Load self-play arrays + manifest; reject a layout mismatch loudly.

    Accept one path (str) or a list of paths.  For each file, asserts
    ``obs_mode == "token"`` and that the layout fields
    ``max_tokens/token_feats/n_tactical/action_size`` match the installed
    ``encode.py`` constants.  Raises ``ValueError`` on mismatch.

    When multiple files are given, each subsequent file's ``game_id`` is
    offset by ``(max game_id so far + 1)`` so games from different files
    never collide (otherwise a game-level val split would leak).

    Returns
    -------
    (arrays, manifest)
        ``arrays`` is the concatenated dict; ``manifest`` is from the first
        file.
    """
    if isinstance(paths, str):
        paths = [paths]
    paths = list(paths)

    all_arrays: list[dict] = []
    first_manifest: dict | None = None
    running_max_gid: int = -1

    for path in paths:
        with open(_manifest_path(path), encoding="utf-8") as f:
            manifest = json.load(f)

        obs_mode = manifest.get("obs_mode", "flat")
        if obs_mode != "token":
            raise ValueError(f"selfplay layout mismatch: obs_mode={obs_mode!r}, expected 'token'")

        mt = manifest.get("max_tokens")
        tf = manifest.get("token_feats")
        nt = manifest.get("n_tactical")
        az = manifest.get("action_size")
        if mt != MAX_TOKENS or tf != TOKEN_FEATS or nt != N_TACTICAL or az != ACTION_SIZE:
            raise ValueError(
                "selfplay layout mismatch: manifest "
                f"max_tokens/token_feats/n_tactical/action_size="
                f"{mt}/{tf}/{nt}/{az} "
                f"!= encode {MAX_TOKENS}/{TOKEN_FEATS}/{N_TACTICAL}/{ACTION_SIZE}; "
                "regenerate the selfplay dataset"
            )

        with np.load(path) as data:
            arrays = {k: data[k] for k in _SELFPLAY_ARRAY_KEYS if k in data.files}

        # Offset game_id so games from different files don't collide.
        offset = running_max_gid + 1
        arrays["game_id"] = arrays["game_id"] + offset
        if len(arrays["game_id"]) > 0:
            running_max_gid = int(arrays["game_id"].max())

        all_arrays.append(arrays)
        if first_manifest is None:
            first_manifest = manifest

    # Concatenate all files along axis 0.
    concatenated: dict = {}
    for key in _SELFPLAY_ARRAY_KEYS:
        parts = [a[key] for a in all_arrays if key in a]
        if parts:
            concatenated[key] = np.concatenate(parts, axis=0)

    return concatenated, first_manifest


def az_train(
    data,
    warm_start: str,
    out: str = "az.zip",
    epochs: int = 10,
    batch: int = 256,
    lr: float = 1e-4,
    c_v: float = 0.5,
    val_frac: float = 0.1,
    seed: int = 0,
    verbose: int = 1,
) -> dict:
    """Warm-start AlphaZero trainer: fine-tune both policy and value heads.

    Loads self-play data, warm-starts from an existing token MaskablePPO net,
    and trains both heads — policy via soft cross-entropy to the recorded visit
    distribution, value via MSE to the recorded game outcome.

    Parameters
    ----------
    data:
        Path (str) or list of paths to self-play ``.npz`` files.
    warm_start:
        Path to a saved token ``MaskablePPO`` ``.zip`` model to load and
        fine-tune.  Do NOT pass a fresh init — that throws away the Phase-1
        oracle weights.
    out:
        Output path for the fine-tuned model ``.zip``.
    epochs:
        Number of training epochs.
    batch:
        Mini-batch size.
    lr:
        Adam learning rate (default 1e-4).
    c_v:
        Value loss coefficient (default 0.5).
    val_frac:
        Fraction of games (game-level split) held out for validation.
    seed:
        RNG seed for the game-level split + epoch shuffle.
    verbose:
        Print per-epoch loss when nonzero.

    Returns
    -------
    dict
        Keys: ``out``, ``val_policy_ce``, ``val_value_mse``, ``n_train``,
        ``n_val``, ``epochs``, ``epoch_losses`` (list of per-epoch mean
        training losses, one entry per epoch).
    """
    import torch  # noqa: PLC0415 — optional [ml] dep
    import torch.nn.functional as F  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.envs.distill import split_by_game  # noqa: PLC0415

    arrays, _ = load_selfplay(data)
    train_idx, val_idx = split_by_game(arrays["game_id"], val_frac, seed)

    # Warm-start: load the existing model (do NOT init a fresh net — that would
    # discard the Phase-1 oracle weights).
    model = MaskablePPO.load(warm_start)
    model.policy.set_training_mode(True)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)

    device = model.device

    # Build per-key obs tensors matching the obs-space keys used at training time.
    obs_t = {
        "tokens": torch.as_tensor(arrays["obs_tokens"].astype(np.float32), device=device),
        "card_ids": torch.as_tensor(arrays["obs_card_ids"].astype(np.float32), device=device),
        "token_mask": torch.as_tensor(arrays["obs_token_mask"].astype(np.float32), device=device),
        "scalars": torch.as_tensor(arrays["obs_scalars"].astype(np.float32), device=device),
    }
    target_t = torch.as_tensor(arrays["policy_target"].astype(np.float32), device=device)
    mask_t = torch.as_tensor(arrays["mask"].astype(bool), device=device)
    value_t = torch.as_tensor(arrays["value_target"].astype(np.float32), device=device)

    rng = np.random.default_rng(seed)
    order = np.asarray(train_idx, dtype=np.int64)

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        rng.shuffle(order)
        total = 0.0
        nb = 0
        for start in range(0, len(order), batch):
            sel = torch.as_tensor(order[start : start + batch], device=device)
            batch_obs = {k: v[sel] for k, v in obs_t.items()}
            mask_b = mask_t[sel]
            target_b = target_t[sel]
            value_b = value_t[sel]

            # Policy head: soft cross-entropy to the visit distribution.
            # logits are masked log-probs from the sb3 masked categorical.
            dist = model.policy.get_distribution(batch_obs, action_masks=mask_b)
            logp = dist.distribution.logits  # [B, ACTION_SIZE] masked log-probs

            # The where-guard prevents 0 * -inf = nan for illegal/unvisited slots.
            loss_pi = (
                -(target_b * torch.where(target_b > 0, logp, torch.zeros_like(logp)))
                .sum(dim=1)
                .mean()
            )

            # Value head: MSE to the game outcome.
            v = model.policy.predict_values(batch_obs)  # [B, 1]
            loss_v = F.mse_loss(v.squeeze(-1), value_b)

            loss = loss_pi + c_v * loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1

        epoch_mean = total / max(nb, 1)
        epoch_losses.append(epoch_mean)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  loss={epoch_mean:.4f}")

    # Val report: held-out games, no grad.
    val_policy_ce = float("nan")
    val_value_mse = float("nan")
    if val_idx:
        model.policy.set_training_mode(False)
        v_sel = torch.as_tensor(np.asarray(val_idx, dtype=np.int64), device=device)
        val_obs = {k: v[v_sel] for k, v in obs_t.items()}
        val_target = target_t[v_sel]
        val_mask = mask_t[v_sel]
        val_value = value_t[v_sel]

        with torch.no_grad():
            dist = model.policy.get_distribution(val_obs, action_masks=val_mask)
            logp = dist.distribution.logits
            val_policy_ce = float(-(val_target * logp).sum(1).mean().item())

            v = model.policy.predict_values(val_obs)
            val_value_mse = float(F.mse_loss(v.squeeze(-1), val_value).item())

    model.save(out)
    return {
        "out": out,
        "val_policy_ce": val_policy_ce,
        "val_value_mse": val_value_mse,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "epochs": epochs,
        "epoch_losses": epoch_losses,
    }
