from __future__ import annotations

import random

from locma.core.actions import Attack, Pass, Summon, Use
from locma.core.cards import CardType

_CREATURE = 0


class RandomBattlePolicy:
    def __init__(self, name: str = "random-battle", seed: int = 0):
        self.name = name
        self._seed = seed
        self._r = random.Random(seed)

    def battle_action(self, view, legal, state=None):
        return self._r.choice(legal)

    def reset(self, seed=None):
        self._r = random.Random(self._seed if seed is None else seed)


class GreedyBattlePolicy:
    def __init__(self, name: str = "greedy-battle"):
        self.name = name

    def battle_action(self, view, legal, state=None):
        attacks = [a for a in legal if isinstance(a, Attack)]
        face = [a for a in attacks if a.target_id == -1]

        # (1) Lethal check: if total available face-attack >= op_health, swing face
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


class ScriptedBattlePolicy:
    """Fixed aggressive battle script (one action per call until Pass)."""

    def __init__(self, name: str = "scripted-battle", seed: int = 0):
        self.name = name
        self._seed = seed
        self._r = random.Random(seed)

    def battle_action(self, view, legal, state=None):
        hand_type = {c.instance_id: c.type for c in view.my_hand}

        green = [
            a
            for a in legal
            if isinstance(a, Use) and hand_type.get(a.item_instance_id) == CardType.GREEN_ITEM
        ]
        if green:
            return self._r.choice(green)

        attacks = [a for a in legal if isinstance(a, Attack)]
        face = [a for a in attacks if a.target_id == -1]
        if face:
            return self._r.choice(face)
        if attacks:  # face not legal (enemy Guards) -> clear a Guard to open it
            return self._r.choice(attacks)

        summons = [a for a in legal if isinstance(a, Summon)]
        if summons:
            return self._r.choice(summons)

        items = [a for a in legal if isinstance(a, Use)]  # remaining red/blue items
        if items:
            return self._r.choice(items)

        return Pass()

    def reset(self, seed=None):
        self._r = random.Random(self._seed if seed is None else seed)


class GroundBattlePolicy:
    """Aggressive ground behaviour: develop the board and swing at the face."""

    def __init__(self, name: str = "ground"):
        self.name = name

    def battle_action(self, view, legal, state=None):
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

    def reset(self, seed=None):
        pass
