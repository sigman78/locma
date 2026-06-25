from locma.policies.base import BattlePolicy, DraftPolicy, Policy


class _Draft:
    name = "d"

    def draft_action(self, view, legal):
        return legal[0]

    def reset(self, seed=None):
        pass


class _Battle:
    name = "b"

    def battle_action(self, view, legal, state=None):
        return legal[0]

    def reset(self, seed=None):
        pass


def test_draft_only_is_draft_policy_not_battle():
    d = _Draft()
    assert isinstance(d, DraftPolicy)
    assert not isinstance(d, BattlePolicy)


def test_battle_action_accepts_optional_state():
    b = _Battle()
    assert isinstance(b, BattlePolicy)
    assert b.battle_action(None, [1, 2]) == 1
    assert b.battle_action(None, [1, 2], state="gs") == 1
