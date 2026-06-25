from __future__ import annotations


class Composer:
    """Builds a Policy from a required Battle Policy + an optional Draft Policy.

    Battle is mandatory (every game mode has a battle); draft is optional and
    absent under Constructed play. With no draft policy, calling ``draft_action``
    is a hard error (an AttributeError on ``None``) — by design, since the engine
    never enters the draft phase in Constructed mode.

    ``name`` is display-only; persistence stores the raw spec string, so the
    registry passes the spec here for tidy table labels.
    """

    def __init__(self, battle, draft=None, name=None):
        self.battle = battle
        self.draft = draft
        if name is not None:
            self.name = name
        elif draft is not None:
            self.name = f"{draft.name}+{battle.name}"
        else:
            self.name = battle.name

    def draft_action(self, view, legal):
        return self.draft.draft_action(view, legal)

    def battle_action(self, view, legal, state=None):
        return self.battle.battle_action(view, legal, state)

    def reset(self, seed=None):
        if self.draft is not None:
            self.draft.reset(seed)
        self.battle.reset(seed)
