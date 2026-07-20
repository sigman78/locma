"""Pointer-style action head as a trainable MaskablePPO policy (E28 gate 2).

E28 gate 1 (worklog 2026-07-19) showed a pointer head — each of the 155
slot-indexed action logits computed FROM the transformer slot tokens of the
cards that action involves — breaks the ~0.37-0.39 BC-agreement cap that the
standard dense ``action_net`` (Linear 64->155) plateaus at, at matched
trainable parameters. This module makes that head trainable under PPO:

- ``PointerActionNet``: drop-in replacement for ``action_net``. Its forward
  still takes ``latent_pi`` (so SB3's distribution plumbing is untouched) and
  reads the per-slot transformer output from a forward hook the policy
  installs on the extractor — the hook fires during ``extract_features``,
  which SB3 always runs earlier in the same forward pass.
- ``PointerMaskablePolicy``: MaskableMultiInputActorCriticPolicy that swaps
  ``action_net`` for a ``PointerActionNet`` after the standard build and
  rebuilds the optimizer. Save/load round-trips through SB3's normal
  machinery (the class is pickled by reference; keep this module importable).

Use from training: ``train_zoo(..., obs_mode="token", pointer_head=True)``.
Bench with the ordinary ``ppo:`` registry spec — ``MaskablePPO.load``
restores the policy class automatically.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from locma.envs.encode import ACTION_SIZE, MAX_TOKENS

NONE_SLOT = MAX_TOKENS  # virtual "no slot" index -> zero token row
USE_LO, USE_HI = 9, 113


def build_action_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-action (src_slot, tgt_slot, family) over the 155 semantic indices.

    Token slots: 0-7 hand, 8-13 my board, 14-19 op board; 20 = none (zeros).
    Families: 0 pass, 1 summon, 2 use, 3 attack.
    """
    src = np.full(ACTION_SIZE, NONE_SLOT, dtype=np.int64)
    tgt = np.full(ACTION_SIZE, NONE_SLOT, dtype=np.int64)
    fam = np.zeros(ACTION_SIZE, dtype=np.int64)
    for i in range(1, 9):
        src[i], fam[i] = i - 1, 1
    for i in range(USE_LO, USE_HI):
        s, tc = divmod(i - USE_LO, 13)
        src[i], fam[i] = s, 2
        if tc < 6:
            tgt[i] = 8 + tc
        elif tc < 12:
            tgt[i] = 14 + (tc - 6)
    for i in range(USE_HI, ACTION_SIZE):
        a, tc = divmod(i - USE_HI, 7)
        src[i], fam[i] = 8 + a, 3
        if tc < 6:
            tgt[i] = 14 + tc
    return src, tgt, fam


class PointerActionNet(nn.Module):
    """logit(a) = MLP([z_src(a), z_tgt(a), latent_pi, family_onehot(a)]).

    ``forward(latent_pi)`` matches the ``action_net`` call signature; the
    slot tokens come from ``slots_ref`` — a zero-argument callable returning
    the (B, MAX_TOKENS, d_model) transformer output of the SAME forward pass
    (the policy wires this to its extractor hook).
    """

    def __init__(self, slots_ref, d_model: int = 64, latent_dim: int = 64, hidden: int = 128):
        super().__init__()
        self._slots_ref = slots_ref
        src, tgt, fam = build_action_table()
        self.register_buffer("src_idx", torch.as_tensor(src), persistent=False)
        self.register_buffer("tgt_idx", torch.as_tensor(tgt), persistent=False)
        self.register_buffer("fam_1h", torch.eye(4)[torch.as_tensor(fam)], persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model + latent_dim + 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # SB3-style init: near-zero final layer -> near-uniform initial policy,
        # matching the gain=0.01 SB3 applies to the standard action_net.
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mlp[-1].weight, gain=0.01)

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        z = self._slots_ref()
        if z is None or z.size(0) != latent_pi.size(0):
            raise RuntimeError(
                "PointerActionNet: slot cache missing or batch-mismatched; "
                "extract_features must run in the same forward pass"
            )
        b = z.size(0)
        zpad = torch.cat([z, z.new_zeros(b, 1, z.size(2))], dim=1)
        srcv = zpad[:, self.src_idx]
        tgtv = zpad[:, self.tgt_idx]
        ctx = latent_pi.unsqueeze(1).expand(-1, ACTION_SIZE, -1)
        fam = self.fam_1h.unsqueeze(0).expand(b, -1, -1)
        return self.mlp(torch.cat([srcv, tgtv, ctx, fam], dim=-1)).squeeze(-1)


class PointerMaskablePolicy(MaskableMultiInputActorCriticPolicy):
    """MaskableMultiInputActorCriticPolicy with a pointer action head.

    Requires a features extractor with a ``transformer`` submodule producing
    (B, MAX_TOKENS, d_model) — i.e. ``TokenSetExtractor``. A forward hook
    caches that output; ``PointerActionNet`` consumes it via closure.
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self._slot_cache: list = [None]
        fe = self.features_extractor
        # The slot source is the transformer (TokenSetExtractor) or, for the
        # E29 transformer-free SlimTokenExtractor, its slot_encoder — both emit
        # (B, MAX_TOKENS, d_model), which is all the pointer gather needs.
        slot_mod = getattr(fe, "transformer", None) or getattr(fe, "slot_encoder", None)
        if slot_mod is None:
            raise ValueError(
                "PointerMaskablePolicy needs an extractor exposing `transformer` or `slot_encoder`"
            )
        pos = getattr(fe, "pos_embed", None)
        if pos is None:  # slim extractor holds pos_embed inside slot_encoder
            pos = fe.slot_encoder.pos_embed
        d_model = pos.shape[-1]

        def _hook(mod, inp, out):
            self._slot_cache[0] = out

        slot_mod.register_forward_hook(_hook)
        self.action_net = PointerActionNet(
            lambda: self._slot_cache[0],
            d_model=d_model,
            latent_dim=self.mlp_extractor.latent_dim_pi,
        )
        # The optimizer built by super()._build() references the old dense
        # action_net's parameters — rebuild it over the final parameter set.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
