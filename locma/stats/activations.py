"""Per-layer activation capture from a loaded MaskablePPO policy (requires [ml]).

Companion to locma.stats.netdiag: this module forwards a frozen probe dataset
(a practicum's observation arrays) through a policy net and returns the hidden
activations per named layer, ready for the numpy metric functions. Torch and
SB3 are imported lazily inside functions, matching the repo's [ml]-optional
convention.

Captured layers
---------------
Flat MlpPolicy (features extractor is a no-op Flatten):
  ``pi_a1``/``pi_a2`` and ``vf_a1``/``vf_a2`` — post-activation outputs of the
  two 64-unit tower layers; ``logits`` — the 155-d action scores.
Token MultiInputPolicy (TokenSetExtractor):
  additionally ``slots`` — the transformer's per-slot output flattened to
  (N, MAX_TOKENS * d_model) — and ``features`` — the extractor's fused head
  output (the input to the towers).

Every entry in the returned kinds dict is "tanh" / "relu" / "linear",
selecting the matching netdiag.unit_health pathology definitions.
"""

from __future__ import annotations

import numpy as np


def practicum_obs(arrays: dict, obs_mode: str):
    """Map practicum storage arrays to the policy's observation format.

    Returns a float32 (N, OBS_SIZE) array for "flat", or a dict keyed by the
    token obs-space names ("tokens", "card_ids", "token_mask", "scalars") for
    "token" — exactly what ``collect_activations`` expects.
    """
    if obs_mode == "token":
        return {
            "tokens": arrays["obs_tokens"].astype(np.float32),
            "card_ids": arrays["obs_card_ids"].astype(np.float32),
            "token_mask": arrays["obs_token_mask"].astype(np.float32),
            "scalars": arrays["obs_scalars"].astype(np.float32),
        }
    return arrays["obs"].astype(np.float32)


def _slice_obs(obs, idx):
    if isinstance(obs, dict):
        return {k: v[idx] for k, v in obs.items()}
    return obs[idx]


def n_examples(obs) -> int:
    if isinstance(obs, dict):
        return len(next(iter(obs.values())))
    return len(obs)


def collect_activations(
    policy,
    obs,
    batch_size: int = 2048,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Forward ``obs`` through ``policy`` capturing per-layer activations.

    Parameters
    ----------
    policy: an ActorCriticPolicy (e.g. ``MaskablePPO.load(...).policy``).
    obs: output of ``practicum_obs`` — (N, D) array or dict of arrays.

    Returns ``(acts, kinds)``: activation matrices (N, width) per layer name,
    and each layer's activation kind for ``netdiag.unit_health``.

    The walk mirrors ActorCriticPolicy.forward exactly — extract_features,
    then the pi/vf towers module-by-module (recording after each non-linear),
    then action_net — so the captured tensors are the net's real activations,
    not a re-implementation.
    """
    import torch  # noqa: PLC0415 — optional [ml] dep
    import torch.nn as nn  # noqa: PLC0415
    from stable_baselines3.common.torch_layers import FlattenExtractor  # noqa: PLC0415

    policy.set_training_mode(False)

    chunks: dict[str, list] = {}
    kinds: dict[str, str] = {}

    def record(name: str, tensor, kind: str) -> None:
        chunks.setdefault(name, []).append(tensor.detach().cpu().numpy().astype(np.float32))
        kinds[name] = kind

    def act_kind(module) -> str:
        if isinstance(module, nn.Tanh):
            return "tanh"
        if isinstance(module, nn.ReLU):
            return "relu"
        return "linear"

    # Token extractor internals: hook the transformer's per-slot output.
    hooks = []
    fe = policy.features_extractor
    slot_out: list = []
    if hasattr(fe, "transformer"):
        hooks.append(
            fe.transformer.register_forward_hook(
                lambda mod, inp, out: slot_out.append(out.detach())
            )
        )
    trivial_extractor = isinstance(fe, FlattenExtractor)

    def walk_tower(name: str, seq, x):
        """Run a Sequential tower, recording after each activation module."""
        i = 0
        for m in seq:
            x = m(x)
            if not isinstance(m, nn.Linear):
                i += 1
                record(f"{name}_a{i}", x, act_kind(m))
        return x

    n = n_examples(obs)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = _slice_obs(obs, slice(start, start + batch_size))
            obs_t, _ = policy.obs_to_tensor(batch)

            slot_out.clear()
            feats = policy.extract_features(obs_t)
            if isinstance(feats, tuple):  # share_features_extractor=False
                pi_feats, vf_feats = feats
            else:
                pi_feats = vf_feats = feats
            if slot_out:
                z = slot_out[0]
                record("slots", z.reshape(z.size(0), -1), "linear")
            if not trivial_extractor:
                # TokenSetExtractor's head ends in ReLU; a Flatten extractor's
                # output IS the observation, so recording it is pure waste.
                record("features", pi_feats, "relu")

            latent_pi = walk_tower("pi", policy.mlp_extractor.policy_net, pi_feats)
            walk_tower("vf", policy.mlp_extractor.value_net, vf_feats)
            record("logits", policy.action_net(latent_pi), "linear")

    for h in hooks:
        h.remove()
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}, kinds


def reinit_clone(policy, seed: int = 0):
    """Deep-copy ``policy`` and re-randomize every parameter — the
    same-architecture random-init baseline for netdiag comparisons.

    Uses each module's own ``reset_parameters`` (PyTorch default init, not
    SB3's orthogonal init — close enough for spectrum-at-init comparisons)
    and re-zeroes TokenSetExtractor's ``pos_embed`` to its init value.
    """
    import copy  # noqa: PLC0415

    import torch  # noqa: PLC0415

    clone = copy.deepcopy(policy)
    torch.manual_seed(seed)
    for m in clone.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        elif hasattr(m, "_reset_parameters"):
            m._reset_parameters()
    fe = clone.features_extractor
    if hasattr(fe, "pos_embed"):
        torch.nn.init.zeros_(fe.pos_embed)
    return clone
