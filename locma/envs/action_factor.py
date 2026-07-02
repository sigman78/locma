"""Torch-free factorization of the Discrete(155) action space into
type -> source -> target, plus derivation of the conditional legality masks
from the flat 155-bool mask. See docs/ppo-autoreg-action-design.md."""

from __future__ import annotations

import numpy as np

PASS, SUMMON, USE, ATTACK = 0, 1, 2, 3
N_TYPE = 4
MAX_SOURCE = 8
MAX_TARGET = 13
ACTION_SIZE = 155

# per type: (base flat index, n_source, n_target)
SEG: tuple[tuple[int, int, int], ...] = (
    (0, 1, 1),  # PASS
    (1, 8, 1),  # SUMMON
    (9, 8, 13),  # USE
    (113, 6, 7),  # ATTACK
)


def decode(idx: int) -> tuple[int, int, int]:
    """Map a flat action index to (type, source, target)."""
    if idx < 1:
        return (PASS, 0, 0)
    if idx < 9:
        return (SUMMON, idx - 1, 0)
    if idx < 113:
        off = idx - 9
        return (USE, off // 13, off % 13)
    off = idx - 113
    return (ATTACK, off // 7, off % 7)


def encode(t: int, s: int, tgt: int) -> int:
    """Inverse of decode: (type, source, target) -> flat action index."""
    base, _n_src, n_tgt = SEG[t]
    return base + s * n_tgt + tgt


def factor_masks(flat_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive conditional legality masks from the flat 155-bool mask.

    Returns (type_mask[4], source_mask[4,8], target_mask[4,8,13]), all bool.
    Non-applicable factors (e.g. a Pass has no real source/target) end up with
    exactly one legal cell, so their conditional log-prob and entropy are 0.
    """
    type_mask = np.zeros(N_TYPE, dtype=bool)
    source_mask = np.zeros((N_TYPE, MAX_SOURCE), dtype=bool)
    target_mask = np.zeros((N_TYPE, MAX_SOURCE, MAX_TARGET), dtype=bool)
    for t, (base, n_src, n_tgt) in enumerate(SEG):
        seg = flat_mask[base : base + n_src * n_tgt].reshape(n_src, n_tgt)
        target_mask[t, :n_src, :n_tgt] = seg
        source_mask[t, :n_src] = seg.any(axis=1)
        type_mask[t] = seg.any()
    return type_mask, source_mask, target_mask
