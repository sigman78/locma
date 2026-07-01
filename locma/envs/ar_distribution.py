"""Autoregressive action head: tensor helpers, head modules, and the
sample/evaluate core. See docs/ppo-autoreg-action-design.md.

Requires the [ml] extra (torch). The torch-free factorization lives in
locma.envs.action_factor."""

from __future__ import annotations

import torch

from locma.envs.action_factor import MAX_SOURCE, MAX_TARGET, N_TYPE, SEG

EMB_DIM = 8

# base / n_target per type, as tensors for vectorized encode
_BASE = torch.tensor([s[0] for s in SEG], dtype=torch.long)
_NTGT = torch.tensor([s[2] for s in SEG], dtype=torch.long)


def decode_batch(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized decode: LongTensor[B] -> (type, source, target)."""
    t = torch.zeros_like(flat)
    t = torch.where(flat >= 1, torch.ones_like(flat), t)
    t = torch.where(flat >= 9, torch.full_like(flat, 2), t)
    t = torch.where(flat >= 113, torch.full_like(flat, 3), t)
    src = torch.zeros_like(flat)
    tgt = torch.zeros_like(flat)
    m_sum = t == 1
    src = torch.where(m_sum, flat - 1, src)
    m_use = t == 2
    src = torch.where(m_use, (flat - 9) // 13, src)
    tgt = torch.where(m_use, (flat - 9) % 13, tgt)
    m_att = t == 3
    src = torch.where(m_att, (flat - 113) // 7, src)
    tgt = torch.where(m_att, (flat - 113) % 7, tgt)
    return t, src, tgt


def encode_batch(t: torch.Tensor, s: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Vectorized encode: (type, source, target) -> flat LongTensor[B]."""
    base = _BASE.to(t.device)[t]
    ntgt = _NTGT.to(t.device)[t]
    return base + s * ntgt + tgt


def factor_grids(flat_masks: torch.Tensor) -> torch.Tensor:
    """Bool[B,155] -> Bool[B,4,8,13]: legal (type,source,target) grid per sample."""
    b = flat_masks.shape[0]
    grids = torch.zeros(
        b, N_TYPE, MAX_SOURCE, MAX_TARGET, dtype=torch.bool, device=flat_masks.device
    )
    for t, (base, n_src, n_tgt) in enumerate(SEG):
        seg = flat_masks[:, base : base + n_src * n_tgt].reshape(b, n_src, n_tgt)
        grids[:, t, :n_src, :n_tgt] = seg
    return grids
