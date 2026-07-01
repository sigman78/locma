"""Autoregressive action head: tensor helpers, head modules, and the
sample/evaluate core. See docs/ppo-autoreg-action-design.md.

Requires the [ml] extra (torch). The torch-free factorization lives in
locma.envs.action_factor."""

from __future__ import annotations

import torch
from sb3_contrib.common.maskable.distributions import MaskableCategorical
from torch import nn

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


class ARHeads(nn.Module):
    """Conditional heads for the autoregressive action distribution.

    type <- z ; source <- (z, emb_type) ; target <- (z, emb_type, emb_source).
    Head widths are the fixed maxima (4 / 8 / 13); masking zeroes out-of-domain
    entries so one source head and one target head serve every action type.
    """

    def __init__(self, latent_dim: int, emb_dim: int = EMB_DIM) -> None:
        super().__init__()
        self.emb_type = nn.Embedding(N_TYPE, emb_dim)
        self.emb_source = nn.Embedding(MAX_SOURCE, emb_dim)
        self.head_type = nn.Linear(latent_dim, N_TYPE)
        self.head_source = nn.Linear(latent_dim + emb_dim, MAX_SOURCE)
        self.head_target = nn.Linear(latent_dim + 2 * emb_dim, MAX_TARGET)
        self.last_head_entropy: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _arange(b: int, device) -> torch.Tensor:
    return torch.arange(b, device=device)


def ar_sample(heads, z, flat_masks, deterministic):
    """Sample type -> source -> target sequentially under derived masks.

    Returns (flat_actions[B] long, log_prob[B]). log_prob is the sum of the
    three conditional log-probs at the chosen values.
    """
    b = z.shape[0]
    idx = _arange(b, z.device)
    grids = factor_grids(flat_masks)  # [B,4,8,13]

    type_mask = grids.any(dim=3).any(dim=2)  # [B,4]
    type_dist = MaskableCategorical(logits=heads.head_type(z), masks=type_mask)
    types = type_dist.probs.argmax(dim=-1) if deterministic else type_dist.sample()

    src_in = torch.cat([z, heads.emb_type(types)], dim=-1)
    src_mask = grids.any(dim=3)[idx, types]  # [B,8]
    src_dist = MaskableCategorical(logits=heads.head_source(src_in), masks=src_mask)
    sources = src_dist.probs.argmax(dim=-1) if deterministic else src_dist.sample()

    tgt_in = torch.cat([z, heads.emb_type(types), heads.emb_source(sources)], dim=-1)
    tgt_mask = grids[idx, types, sources]  # [B,13]
    tgt_dist = MaskableCategorical(logits=heads.head_target(tgt_in), masks=tgt_mask)
    targets = tgt_dist.probs.argmax(dim=-1) if deterministic else tgt_dist.sample()

    log_prob = type_dist.log_prob(types) + src_dist.log_prob(sources) + tgt_dist.log_prob(targets)
    return encode_batch(types, sources, targets), log_prob


def ar_evaluate(heads, z, flat_masks, actions):
    """Teacher-forced scoring of given flat actions.

    Returns (log_prob[B], entropy[B]) where entropy is the sum of the three
    conditional entropies along the visited prefix.
    """
    b = z.shape[0]
    idx = _arange(b, z.device)
    grids = factor_grids(flat_masks)
    types, sources, targets = decode_batch(actions)

    type_mask = grids.any(dim=3).any(dim=2)
    type_dist = MaskableCategorical(logits=heads.head_type(z), masks=type_mask)

    src_in = torch.cat([z, heads.emb_type(types)], dim=-1)
    src_mask = grids.any(dim=3)[idx, types]
    src_dist = MaskableCategorical(logits=heads.head_source(src_in), masks=src_mask)

    tgt_in = torch.cat([z, heads.emb_type(types), heads.emb_source(sources)], dim=-1)
    tgt_mask = grids[idx, types, sources]
    tgt_dist = MaskableCategorical(logits=heads.head_target(tgt_in), masks=tgt_mask)

    log_prob = type_dist.log_prob(types) + src_dist.log_prob(sources) + tgt_dist.log_prob(targets)
    ent_type = type_dist.entropy()
    ent_src = src_dist.entropy()
    ent_tgt = tgt_dist.entropy()
    heads.last_head_entropy = (
        float(ent_type.mean().detach()),
        float(ent_src.mean().detach()),
        float(ent_tgt.mean().detach()),
    )
    return log_prob, ent_type + ent_src + ent_tgt
