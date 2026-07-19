"""Ground-truth concept labels at battle decision points (E27 concept probes).

Each labeler is a deterministic function of the FULL game state at an
``on_pre_step`` decision (labels may use hidden information — they are
ground truth about the position, not features the net could see). Labels are
probed per layer against the SAME probe on the raw observation; a concept is
only "computed by the net" where some layer beats that raw-obs control.

Concepts
--------
Binary (probe_classify, AUC):
  - ``lethal_now``: the mover has an engine-verified forced win this turn
    (exhaustive ``lguard.find_lethal`` DFS; -1 when the node cap was hit
    without exhausting — absence not established, row excluded from probes).
  - ``opp_threat_lethal``: threat arithmetic — the opponent's current board
    attack, after chewing through the mover's guard wall, meets or exceeds
    the mover's health. "If I do nothing, I die to the board next turn."
Continuous (probe_regression, R^2) — mostly near-linear sanity controls:
  - ``hp_diff``: mover health minus opponent health.
  - ``own_mana``: mover's available mana this turn.
  - ``board_atk_diff``: total board attack, mover minus opponent.
  - ``my_guards``: number of guard creatures on the mover's board.
"""

from __future__ import annotations

from locma.policies.lguard import find_lethal

CONCEPT_KEYS = (
    "lethal_now",
    "opp_threat_lethal",
    "hp_diff",
    "own_mana",
    "board_atk_diff",
    "my_guards",
)

BINARY_CONCEPTS = ("lethal_now", "opp_threat_lethal")
CONTINUOUS_CONCEPTS = ("hp_diff", "own_mana", "board_atk_diff", "my_guards")


def _opp_threat_lethal(me, opp) -> float:
    """1.0 iff the opponent's standing board can kill through the guard wall.

    Deliberately simple threat arithmetic (no ward/lethal keywords, no item
    burn): opponent creatures must chew through the mover's guard defense
    before hitting face, so the face damage available next turn is
    ``max(0, opp_attack_total - my_guard_defense)`` under the crude model of
    attack points spent 1:1 on guard defense. A well-defined, deterministic
    function of the visible boards that is NOT a linear readout of either.
    """
    opp_atk = sum(c.attack for c in opp.board)
    guard_def = sum(c.defense for c in me.board if c.has("G"))
    return 1.0 if opp_atk - guard_def >= me.health else 0.0


def concept_labels(gs, node_cap: int = 3000) -> dict[str, float]:
    """All concept labels for the mover (``gs.current``) at this decision."""
    seat = gs.current
    me = gs.players[seat]
    opp = gs.players[1 - seat]

    line, exhausted = find_lethal(gs, node_cap=node_cap)
    if line is not None:
        lethal = 1.0
    elif exhausted:
        lethal = 0.0
    else:
        lethal = -1.0  # cap hit: absence not established

    return {
        "lethal_now": lethal,
        "opp_threat_lethal": _opp_threat_lethal(me, opp),
        "hp_diff": float(me.health - opp.health),
        "own_mana": float(me.mana),
        "board_atk_diff": float(sum(c.attack for c in me.board) - sum(c.attack for c in opp.board)),
        "my_guards": float(sum(1 for c in me.board if c.has("G"))),
    }
