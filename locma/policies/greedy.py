from __future__ import annotations
from locma.core.actions import Summon, Attack, Use, Pass
from locma.core.cards import ABILITY_ORDER


def _kw_count(abilities: str) -> int:
    return sum(1 for ch in abilities if ch != "-")


def _score(cv) -> float:
    base = cv.attack + cv.defense + 0.5 * _kw_count(cv.abilities)
    if cv.type != 0:  # items slightly deprioritized in draft
        base -= 1.0
    return base


class GreedyPolicy:
    def __init__(self, name: str = "greedy"):
        self.name = name

    def draft_action(self, view, legal):
        scores = [_score(cv) for cv in view.offered]
        return max(legal, key=lambda i: scores[i])

    def battle_action(self, view, legal):
        attacks = [a for a in legal if isinstance(a, Attack)]
        face = [a for a in attacks if a.target_id == -1]

        # (1) Lethal check: if total available face-attack >= op_health, attack face
        total_face = 0
        for a in face:
            for c in view.my_board:
                if c.instance_id == a.attacker_id:
                    total_face += c.attack
        if face and total_face >= view.op_health:
            return face[0]

        # (2) Summon the most expensive affordable creature
        summons = [a for a in legal if isinstance(a, Summon)]
        if summons:
            def cost_of(a):
                for c in view.my_hand:
                    if c.instance_id == a.card_instance_id:
                        return c.cost
                return 0
            return max(summons, key=cost_of)

        # (3) Attack into enemy creatures, else attack face
        creature_attacks = [a for a in attacks if a.target_id != -1]
        if creature_attacks:
            return creature_attacks[0]
        if face:
            return face[0]

        # (4) Pass
        return Pass()

    def reset(self, seed=None):
        pass
