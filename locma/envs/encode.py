"""Action/observation encoding for the Gymnasium BattleEnv.

Observation layout (OBS_SIZE = 146 floats):
  - 6 scalar features: turn, me_health, me_mana, op_health, op_hand_count, (reserved 0.0)
  - 6 * 7 = 42 floats for my_board  (6 slots, 7 features each)
  - 6 * 7 = 42 floats for op_board  (6 slots, 7 features each)
  - 8 * 7 = 56 floats for my_hand   (8 slots, 7 features each)
  Total: 6 + 42 + 42 + 56 = 146

Per-card 7-feature block:
  [present, cost, attack, defense, hasGuard, hasLethal, hasWard]
"""

from __future__ import annotations

import numpy as np

from locma.core.actions import Pass

ACTION_SIZE: int = 64  # max canonical legal actions considered per decision
OBS_SIZE: int = 6 + 6 * 7 + 6 * 7 + 8 * 7  # 6 + 42 + 42 + 56 = 146


def _card_feats(seq, n: int) -> list[float]:
    """Encode `n` card slots from `seq`; missing slots are zero-padded."""
    out: list[float] = []
    for i in range(n):
        if i < len(seq):
            c = seq[i]
            out += [
                1.0,
                float(c.cost),
                float(c.attack),
                float(c.defense),
                float("G" in c.abilities),
                float("L" in c.abilities),
                float("W" in c.abilities),
            ]
        else:
            out += [0.0] * 7
    return out


def encode_battle(view) -> np.ndarray:
    """Encode a BattleView into a fixed-length float32 observation vector."""
    vec: list[float] = [
        float(view.turn),
        float(view.me_health),
        float(view.me_mana),
        float(view.op_health),
        float(view.op_hand_count),
        0.0,  # reserved padding to reach 6 scalars
    ]
    vec += _card_feats(view.my_board, 6)
    vec += _card_feats(view.op_board, 6)
    vec += _card_feats(view.my_hand, 8)
    arr = np.asarray(vec, dtype=np.float32)
    assert len(arr) == OBS_SIZE, f"encode_battle length mismatch: {len(arr)} != {OBS_SIZE}"
    return arr


def action_mask(legal) -> np.ndarray:
    """Return a boolean mask of length ACTION_SIZE; first len(legal) entries are True."""
    m = np.zeros(ACTION_SIZE, dtype=bool)
    m[: min(len(legal), ACTION_SIZE)] = True
    return m


def index_to_action(idx: int, legal):
    """Map a discrete index to a legal action; out-of-range returns Pass()."""
    if 0 <= idx < len(legal):
        return legal[idx]
    return Pass()
