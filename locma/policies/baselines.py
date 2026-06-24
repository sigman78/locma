"""Baseline tournament policies.

Each baseline pairs a simple draft heuristic with a shared "ground" battle
behaviour: develop the board and swing at the enemy face, falling back to
clearing Guards when the face is not a legal target (``battle_legal`` only
offers face attacks while the enemy has no Guards, so the retarget is
automatic). The engine calls :meth:`battle_action` repeatedly until it
returns ``Pass``, so returning one action per call still empties the hand
and attacks with every creature over the course of a turn.
"""

from __future__ import annotations

from locma.core.actions import Attack, Pass, Summon

_CREATURE = 0  # CardView.type for creatures


class GroundBattlePolicy:
    """Aggressive ground behaviour shared by the baseline policies.

    Subclasses override :meth:`draft_action` to supply a draft heuristic.
    """

    def __init__(self, name: str = "ground"):
        self.name = name

    def battle_action(self, view, legal):
        attacks = [a for a in legal if isinstance(a, Attack)]
        face = [a for a in attacks if a.target_id == -1]
        if face:  # prioritise attacking face
            return face[0]
        if attacks:  # face not legal (enemy Guards) -> clear a Guard
            return attacks[0]
        summons = [a for a in legal if isinstance(a, Summon)]
        if summons:  # develop the board with what is in hand
            return summons[0]
        return Pass()

    def draft_action(self, view, legal):
        return legal[0]

    def reset(self, seed=None):
        pass


class MaxGuardDraftPolicy(GroundBattlePolicy):
    """Draft Guard creatures above all else; battle on the ground."""

    def __init__(self, name: str = "max-guard"):
        super().__init__(name)

    def draft_action(self, view, legal):
        def key(i):
            cv = view.offered[i]
            is_creature = cv.type == _CREATURE
            has_guard = is_creature and "G" in cv.abilities
            return (has_guard, is_creature, cv.attack + cv.defense)

        return max(legal, key=key)


class MaxAttackDraftPolicy(GroundBattlePolicy):
    """Draft the highest-attack creature; battle on the ground."""

    def __init__(self, name: str = "max-attack"):
        super().__init__(name)

    def draft_action(self, view, legal):
        def key(i):
            cv = view.offered[i]
            is_creature = cv.type == _CREATURE
            return (is_creature, cv.attack, cv.defense)

        return max(legal, key=key)
