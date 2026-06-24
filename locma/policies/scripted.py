from __future__ import annotations

import random

from locma.core.actions import Attack, Pass, Summon, Use
from locma.core.cards import CardType


class ScriptedPolicy:
    """Random draft + a fixed aggressive battle script.

    Battle priority (each call returns one action; the engine calls until
    ``Pass``):

    1. use green items on own creatures,
    2. attack the enemy face — or a Guard first, when ``battle_legal`` offers no
       face attack, to open the face,
    3. summon creatures,
    4. use any remaining (red/blue) items,
    5. pass.

    Within a tier the action (and therefore its target) is chosen at random.
    """

    def __init__(self, name: str = "scripted", seed: int = 0):
        self.name = name
        self._seed = seed
        self._r = random.Random(seed)

    def draft_action(self, view, legal):
        return self._r.choice(legal)

    def battle_action(self, view, legal):
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
