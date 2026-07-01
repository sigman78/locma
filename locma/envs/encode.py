"""Action/observation encoding for the Gymnasium BattleEnv.

Fixed SEMANTIC action space (slot-indexed, ACTION_SIZE=155): each index has a
stable meaning and the mask flags which concrete actions are legal. Observation
(OBS_SIZE=308): 8 scalars + 20 card slots (hand 8 + my board 6 + op board 6) x 15
features each.

Tokenized observation (PPO2 / additive — do NOT remove flat path):
  tokens (20,17) + card_ids (20,) + token_mask (20,) + scalars (13,).
  See docs/ppo2-tokenized-obs-design.md §1 for full layout.
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

# ---------------------------------------------------------------------------
# Tokenized-observation constants (PPO2 additive)
# ---------------------------------------------------------------------------

# Slot layout: 0..7 = my_hand, 8..13 = my_board, 14..19 = op_board
MAX_TOKENS: int = MAX_HAND + MAX_BOARD + MAX_BOARD  # 20

# Per-token numeric features (zone 3 + type 4 + cost/atk/def 3 + abilities 6 + ready 1)
TOKEN_FEATS: int = 3 + 4 + 3 + N_ABILITY + 1  # 17

# Card-id vocabulary: 0 = PAD, 1..160 = real card ids
NUM_CARDS: int = 160

# Tactical scalar count (see encode_battle_tokens docstring for full list)
N_TACTICAL: int = 13

# V1 tactical scalar count: V0's 13 + 5 symmetric-threat scalars (variant="v1").
N_TACTICAL_V1: int = 18


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


# ---------------------------------------------------------------------------
# Tokenized observation encoder (PPO2 — additive; flat path unchanged above)
# ---------------------------------------------------------------------------

# Guard ability sits at index 3 in the BCDGLW ability string.
_GUARD_IDX: int = 3


def _token_row(card, zone_idx: int, *, on_board: bool) -> list:
    """Build a single 17-element feature row for one card token.

    Layout: zone_onehot(3) | type_onehot(4) | cost,atk,def(3) | abilities(6) | ready(1)
    """
    zone = [0.0, 0.0, 0.0]
    zone[zone_idx] = 1.0

    t = card.type
    type_oh = [float(t == i) for i in range(4)]

    ab = card.abilities
    abilities = [float(ab[i] != "-") for i in range(N_ABILITY)]

    ready = float(on_board and card.can_attack and not card.has_attacked)

    cad = [float(card.cost), float(card.attack), float(card.defense)]
    return zone + type_oh + cad + abilities + [ready]


def encode_battle_tokens(view, variant: str = "v0") -> dict:
    """Encode a BattleView into the tokenized PPO2 observation dict.

    Returns a dict with four float32 numpy arrays:
      - ``tokens``     shape (20, 17): per-card numeric features; zeros for pads
      - ``card_ids``   shape (20,):    card_id (1..160); 0 for pads
      - ``token_mask`` shape (20,):    1 for real cards, 0 for pads
      - ``scalars``    shape (13,) for variant="v0", (18,) for variant="v1":
                       tactical scalars (see below)

    Slot order: 0..7 my_hand / 8..13 my_board / 14..19 op_board.

    Tactical scalar order (N_TACTICAL=13):
      0  turn
      1  me_health
      2  op_health
      3  me_mana
      4  summonable_count   (hand cards with cost <= me_mana)
      5  op_hand_count
      6  my_board_count
      7  op_board_count
      8  opp_guard_count    (op-board creatures with G ability)
      9  my_total_attack    (sum of attack over my_board)
      10 my_total_defense   (sum of defense over my_board)
      11 reachable_face_damage  (0 if any opp guard else sum ready-attacker attack)
      12 lethal_available   (1.0 if reachable_face_damage >= op_health else 0.0)

    variant="v1" appends 5 symmetric-threat scalars (N_TACTICAL_V1=18) that give
    the net awareness of the *opponent's* threat back at me (V0 only encodes my
    own guard/attack/reachable):
      13 my_guard_count    (my-board creatures with G ability)
      14 op_total_attack   (sum of attack over op_board)
      15 op_reachable      (0 if any my guard else sum op ready-attacker attack)
      16 exposed_to_lethal (1.0 if op_reachable >= me_health else 0.0)
      17 card_advantage    ((my_hand+my_board) - (op_hand+op_board) card counts)
    """
    tokens = np.zeros((MAX_TOKENS, TOKEN_FEATS), dtype=np.float32)
    card_ids = np.zeros(MAX_TOKENS, dtype=np.float32)
    token_mask = np.zeros(MAX_TOKENS, dtype=np.float32)

    # --- fill hand slots (zone 0, on_board=False) ---------------------------
    for i, card in enumerate(view.my_hand):
        if i >= MAX_HAND:
            break
        tokens[i] = _token_row(card, 0, on_board=False)
        card_ids[i] = float(card.card_id)
        token_mask[i] = 1.0

    # --- fill my_board slots (zone 1, on_board=True) ------------------------
    for i, card in enumerate(view.my_board):
        if i >= MAX_BOARD:
            break
        slot = MAX_HAND + i  # 8..13
        tokens[slot] = _token_row(card, 1, on_board=True)
        card_ids[slot] = float(card.card_id)
        token_mask[slot] = 1.0

    # --- fill op_board slots (zone 2, on_board=True) ------------------------
    for i, card in enumerate(view.op_board):
        if i >= MAX_BOARD:
            break
        slot = MAX_HAND + MAX_BOARD + i  # 14..19
        tokens[slot] = _token_row(card, 2, on_board=True)
        card_ids[slot] = float(card.card_id)
        token_mask[slot] = 1.0

    # --- compute tactical scalars -------------------------------------------
    opp_guard_count = sum(1 for c in view.op_board if c.abilities[_GUARD_IDX] != "-")
    my_total_attack = sum(float(c.attack) for c in view.my_board)
    my_total_defense = sum(float(c.defense) for c in view.my_board)
    summonable_count = sum(1 for c in view.my_hand if c.cost <= view.me_mana)

    if opp_guard_count > 0:
        reachable_face_damage = 0.0
    else:
        reachable_face_damage = sum(
            float(c.attack) for c in view.my_board if c.can_attack and not c.has_attacked
        )

    lethal_available = 1.0 if reachable_face_damage >= view.op_health else 0.0

    scalars = np.array(
        [
            float(view.turn),
            float(view.me_health),
            float(view.op_health),
            float(view.me_mana),
            float(summonable_count),
            float(view.op_hand_count),
            float(len(view.my_board)),
            float(len(view.op_board)),
            float(opp_guard_count),
            my_total_attack,
            my_total_defense,
            reachable_face_damage,
            lethal_available,
        ],
        dtype=np.float32,
    )

    if variant == "v1":
        my_guard_count = sum(1 for c in view.my_board if c.abilities[_GUARD_IDX] != "-")
        op_total_attack = sum(float(c.attack) for c in view.op_board)
        if my_guard_count > 0:
            op_reachable = 0.0
        else:
            op_reachable = sum(
                float(c.attack) for c in view.op_board if c.can_attack and not c.has_attacked
            )
        exposed_to_lethal = 1.0 if op_reachable >= view.me_health else 0.0
        card_advantage = float(
            (len(view.my_hand) + len(view.my_board)) - (view.op_hand_count + len(view.op_board))
        )
        scalars = np.concatenate(
            [
                scalars,
                np.array(
                    [
                        my_guard_count,
                        op_total_attack,
                        op_reachable,
                        exposed_to_lethal,
                        card_advantage,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    return {
        "tokens": tokens,
        "card_ids": card_ids,
        "token_mask": token_mask,
        "scalars": scalars,
    }


def token_obs_space(variant: str = "v0"):
    """Return the gymnasium spaces.Dict for the tokenized PPO2 observation.

    gymnasium is imported lazily so this module stays import-safe without ML deps.
    """
    from gymnasium import spaces  # noqa: PLC0415

    n_scalar = N_TACTICAL_V1 if variant == "v1" else N_TACTICAL

    return spaces.Dict(
        {
            "tokens": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(MAX_TOKENS, TOKEN_FEATS),
                dtype=np.float32,
            ),
            "card_ids": spaces.Box(
                low=0,
                high=NUM_CARDS,
                shape=(MAX_TOKENS,),
                dtype=np.float32,
            ),
            "token_mask": spaces.Box(
                low=0,
                high=1,
                shape=(MAX_TOKENS,),
                dtype=np.float32,
            ),
            "scalars": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_scalar,),
                dtype=np.float32,
            ),
        }
    )
