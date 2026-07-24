"""E38 afterstate consequence columns — per-action, engine-computed, FAIR.

For each of the acting seat's legal actions, simulate that ONE action on a clone
of the decision state and read off what it did. The result is a
``(ACTION_SIZE, N_CONSEQ)`` block indexed by the same fixed semantic action
indices as ``encode.action_mask`` — so it lines up with the pointer head, which
already computes ``logit(a) = MLP([z_src(a), z_tgt(a), latent_pi, fam(a)])`` and
can simply take these columns as extra per-action inputs. Illegal actions are
zeros.

Pre-registration and rationale: ``docs/e38-afterstate-columns-design.md``. In
short, three of the program's live residuals are per-action quantities the trunk
demonstrably does not compute (E27: it computes exactly one concept beyond its
input) — trade value (E33: dphi 29% too low), overkill (E33: +15%, an
attacker/target mismatch), and missed lethal (flat 0.082-0.105 through E28/E29/
E36 and all four E37 doses).

FAIRNESS — this module must never read hidden information (``CONTEXT.md``: the
forward model is a perfect-information view, so a policy that reads it is
"cheating"). E38 is fair on exactly the argument that makes E26's ``lguard``
fair:

  * only OUR OWN candidate actions are simulated — the opponent never moves, so
    their hidden hand/deck is never consulted;
  * features read only public or own-known afterstate: board contents/stats,
    both heroes' health, our own mana;
  * SELF-LEAK GUARD: if an action draws us cards, the drawn cards are NOT read.
    A player knows their deck's contents but not its shuffled order (the same
    reasoning behind ``determinize(reshuffle_own=True)``), so reading a drawn
    card would leak our own future. No column touches hand contents.
  * ``exposes_lethal`` uses the opponent's VISIBLE board only — the identical
    computation the shipped ``v1`` scalar variant already performs.

``Pass`` (index 0) is always zeros: ``apply_battle(Pass)`` fuses the turn
transition (``docs/ideas.md`` — it flips the seat and draws for the opponent), so
"the state after Pass" is the opponent's turn, not our end-of-turn state. Like
``lguard``, we never simulate it.
"""

from __future__ import annotations

import numpy as np

from locma.core.actions import Attack, Pass
from locma.envs.encode import ACTION_SIZE, sem_index

# Column order — keep in sync with the design doc's table and CONSEQ_COLUMNS.
CONSEQ_COLUMNS: tuple[str, ...] = (
    "dphi",  # 0 board-power differential delta (E33's quantity)
    "dmg_face",  # 1 damage dealt to the enemy hero
    "d_my_health",  # 2 own health change (heals / self-damage)
    "n_kills",  # 3 opponent creatures removed
    "n_losses",  # 4 own creatures removed
    "overkill",  # 5 attack wasted beyond target defense (creature-attacks)
    "mana_after",  # 6 own mana remaining
    "wins_now",  # 7 this action ends the game in our favour
    "enables_lethal",  # 8 an exhaustive own-turn lethal exists AFTER this action
    "exposes_lethal",  # 9 op's visible board can kill us next turn
)
N_CONSEQ: int = len(CONSEQ_COLUMNS)

# Guard sits at index 3 in the BCDGLW ability string (mirrors encode._GUARD_IDX).
_GUARD_IDX = 3

# node_cap for the per-action lethal probe. Lower than LethalGuardBattlePolicy's
# 3000 because this runs once per LEGAL ACTION rather than once per turn; a
# cap-hit yields (None, False) = "absence not established", recorded as 0.0 and
# counted in the stats so the gate can report how often it happened.
LETHAL_NODE_CAP: int = 600


def _power(board) -> int:
    return sum(c.attack + c.defense for c in board)


def _phi(gs, seat: int) -> int:
    """Board-power differential from ``seat``'s perspective (E33/E34's potential)."""
    return _power(gs.players[seat].board) - _power(gs.players[1 - seat].board)


def _op_reachable(gs, seat: int) -> float:
    """Damage the opponent's VISIBLE board can send at our face next turn.

    Sums the whole opponent board, not just currently-ready creatures: their
    ``start_turn`` refreshes every creature, so all of them will be able to act
    when they next move (the same reasoning as the ``v1`` scalar ``op_reachable``).
    Zero if we hold a Guard.
    """
    mine, theirs = gs.players[seat].board, gs.players[1 - seat].board
    if any(c.abilities[_GUARD_IDX] != "-" for c in mine):
        return 0.0
    return float(sum(c.attack for c in theirs))


def _overkill(gs, action, seat: int) -> float:
    """Attack wasted beyond the target's defense, for a creature-attack only."""
    if not isinstance(action, Attack) or action.target_id == -1:
        return 0.0
    atk = next((c for c in gs.players[seat].board if c.instance_id == action.attacker_id), None)
    tgt = next((c for c in gs.players[1 - seat].board if c.instance_id == action.target_id), None)
    if atk is None or tgt is None:
        return 0.0
    return float(max(0, atk.attack - tgt.defense))


def conseq_features(gs, legal=None, *, stats: dict | None = None) -> np.ndarray:
    """Per-action consequence block for ``gs.current``'s legal actions.

    Returns ``(ACTION_SIZE, N_CONSEQ)`` float32; zeros for illegal actions, for
    actions the semantic space cannot address, and for ``Pass``. Never mutates
    ``gs`` (all simulation runs on ``_clone_battle`` copies).

    ``stats``, if given, is updated in place with ``actions`` (simulated),
    ``lethal_probes`` (DFS calls that passed the cheap filter) and
    ``lethal_capped`` (probes that hit ``LETHAL_NODE_CAP``) — the cost/validity
    bookkeeping the gate reports.
    """
    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.engine import make_battle_view  # noqa: PLC0415
    from locma.core.state import Phase  # noqa: PLC0415
    from locma.policies.lguard import find_lethal  # noqa: PLC0415
    from locma.policies.mcts import _clone_battle  # noqa: PLC0415

    out = np.zeros((ACTION_SIZE, N_CONSEQ), dtype=np.float32)
    seat = gs.current
    if legal is None:
        legal = list(battlemod.battle_legal(gs))
    view = make_battle_view(gs)

    me0, op0 = gs.players[seat], gs.players[1 - seat]
    phi0 = _phi(gs, seat)
    my_hp0, op_hp0 = me0.health, op0.health
    n_op0, n_my0 = len(op0.board), len(me0.board)

    for action in legal:
        if isinstance(action, Pass):
            continue  # apply_battle(Pass) fuses end_turn — never simulate it
        idx = sem_index(view, action)
        if idx is None or idx >= ACTION_SIZE:
            continue

        ok = _overkill(gs, action, seat)
        sim = _clone_battle(gs)
        battlemod.apply_battle(sim, action)
        if stats is not None:
            stats["actions"] = stats.get("actions", 0) + 1

        me1, op1 = sim.players[seat], sim.players[1 - seat]
        ended = sim.phase == Phase.ENDED
        row = out[idx]
        row[0] = float(_phi(sim, seat) - phi0)
        row[1] = float(op_hp0 - op1.health)
        row[2] = float(me1.health - my_hp0)
        row[3] = float(max(0, n_op0 - len(op1.board)))
        row[4] = float(max(0, n_my0 - len(me1.board)))
        row[5] = ok
        row[6] = float(me1.mana)
        row[7] = 1.0 if ended and sim.winner == seat else 0.0

        if not ended:
            # Cheap necessary condition before the DFS: a lethal needs at least
            # op_health worth of reachable own attack. Most states fail it, so
            # the exhaustive probe runs rarely.
            reachable = (
                0.0
                if any(c.abilities[_GUARD_IDX] != "-" for c in op1.board)
                else sum(float(c.attack) for c in me1.board if c.can_attack and not c.has_attacked)
            )
            if reachable >= op1.health:
                if stats is not None:
                    stats["lethal_probes"] = stats.get("lethal_probes", 0) + 1
                line, exhausted = find_lethal(sim, node_cap=LETHAL_NODE_CAP)
                if line is not None:
                    row[8] = 1.0
                elif not exhausted and stats is not None:
                    stats["lethal_capped"] = stats.get("lethal_capped", 0) + 1
            row[9] = 1.0 if _op_reachable(sim, seat) >= me1.health else 0.0

    return out
