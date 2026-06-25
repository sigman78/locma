"""Action/observation encoding for the Gymnasium BattleEnv.

Fixed SEMANTIC action space (slot-indexed, ACTION_SIZE=155): each index has a
stable meaning and the mask flags which concrete actions are legal. Observation
(OBS_SIZE=308): 8 scalars + 20 card slots (hand 8 + my board 6 + op board 6) x 15
features each.
"""

from __future__ import annotations

import numpy as np

from locma.core.actions import Attack, Pass, Summon, Use

MAX_HAND = 8
MAX_BOARD = 6
N_ABILITY = 6  # BCDGLW
CARD_FEATS = 5 + 3 + N_ABILITY + 1  # 15
N_SCALARS = 8

ACTION_SIZE: int = 1 + MAX_HAND + MAX_HAND * 13 + MAX_BOARD * 7  # 155
OBS_SIZE: int = N_SCALARS + (MAX_HAND + MAX_BOARD + MAX_BOARD) * CARD_FEATS  # 308


def _card_block(c, *, on_board: bool) -> list[float]:
    t = c.type
    ab = c.abilities
    return [
        1.0,
        float(t == 0),  # creature
        float(t == 1),  # green item
        float(t == 2),  # red item
        float(t == 3),  # blue item
        float(c.cost),
        float(c.attack),
        float(c.defense),
        *(float(ab[i] != "-") for i in range(N_ABILITY)),
        float(on_board and c.can_attack and not c.has_attacked),
    ]


def encode_battle(view) -> np.ndarray:
    """Encode a BattleView into a fixed-length float32 observation vector."""
    vec: list[float] = [
        float(view.turn),
        float(view.me_health),
        float(view.me_mana),
        float(view.op_health),
        float(view.op_hand_count),
        float(len(view.my_board)),
        float(len(view.op_board)),
        float(len(view.my_hand)),
    ]

    def pad(seq, n, *, on_board):
        out: list[float] = []
        for i in range(n):
            if i < len(seq):
                out += _card_block(seq[i], on_board=on_board)
            else:
                out += [0.0] * CARD_FEATS
        return out

    vec += pad(view.my_hand, MAX_HAND, on_board=False)
    vec += pad(view.my_board, MAX_BOARD, on_board=True)
    # op_board readiness reflects the opponent's last turn (it only refreshes on
    # their start_turn), so the "ready" bit here means "didn't attack last turn"
    # rather than anything actionable — kept for layout symmetry, low signal.
    vec += pad(view.op_board, MAX_BOARD, on_board=True)
    arr = np.asarray(vec, dtype=np.float32)
    assert len(arr) == OBS_SIZE, f"encode_battle length {len(arr)} != {OBS_SIZE}"
    return arr


def _slot(seq, iid):
    for i, c in enumerate(seq):
        if c.instance_id == iid:
            return i
    return None


def _use_target_code(view, target_id):
    if target_id == -1:
        return 12
    j = _slot(view.my_board, target_id)
    if j is not None:
        return j
    j = _slot(view.op_board, target_id)
    if j is not None:
        return 6 + j
    return None


def _attack_target_code(view, target_id):
    if target_id == -1:
        return 6
    j = _slot(view.op_board, target_id)
    if j is not None:
        return j
    return None


def sem_index(view, action):
    """Map a concrete Action to its fixed semantic slot index (or None)."""
    if isinstance(action, Pass):
        return 0
    if isinstance(action, Summon):
        s = _slot(view.my_hand, action.card_instance_id)
        return 1 + s if s is not None and s < MAX_HAND else None
    if isinstance(action, Use):
        s = _slot(view.my_hand, action.item_instance_id)
        if s is None or s >= MAX_HAND:
            return None
        tc = _use_target_code(view, action.target_id)
        return 9 + s * 13 + tc if tc is not None else None
    if isinstance(action, Attack):
        a = _slot(view.my_board, action.attacker_id)
        if a is None or a >= MAX_BOARD:
            return None
        tc = _attack_target_code(view, action.target_id)
        return 113 + a * 7 + tc if tc is not None else None
    return None


def _action_map(view, legal):
    m = {}
    for a in legal:
        idx = sem_index(view, a)
        if idx is not None:
            m[idx] = a
    return m


def action_mask(view, legal) -> np.ndarray:
    """Boolean mask of length ACTION_SIZE; True exactly at legal semantic indices."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for idx in _action_map(view, legal):
        mask[idx] = True
    return mask


def index_to_action(view, legal, idx):
    """Map a semantic index back to a concrete legal Action; unknown -> Pass()."""
    return _action_map(view, legal).get(int(idx), Pass())
