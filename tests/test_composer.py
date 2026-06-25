import pytest

from locma.policies.composer import Composer


class _Draft:
    name = "gd"

    def __init__(self):
        self.reset_seed = "unset"

    def draft_action(self, view, legal):
        return legal[-1]

    def reset(self, seed=None):
        self.reset_seed = seed


class _Battle:
    name = "gb"

    def __init__(self):
        self.reset_seed = "unset"

    def battle_action(self, view, legal, state=None):
        return ("battle", state, legal[0])

    def reset(self, seed=None):
        self.reset_seed = seed


def test_composer_delegates_and_default_name():
    c = Composer(_Battle(), _Draft())
    assert c.name == "gd+gb"
    assert c.draft_action(None, [0, 1, 2]) == 2
    assert c.battle_action(None, [9], state="GS") == ("battle", "GS", 9)


def test_composer_explicit_name_and_reset_threads_seed():
    b, d = _Battle(), _Draft()
    c = Composer(b, d, name="greedy")
    c.reset(7)
    assert c.name == "greedy"
    assert b.reset_seed == 7 and d.reset_seed == 7


def test_composer_battle_only_constructed_mode():
    b = _Battle()
    c = Composer(b)  # no draft
    assert c.name == "gb"
    c.reset(3)  # must not crash with no draft
    assert b.reset_seed == 3
    with pytest.raises(AttributeError):
        c.draft_action(None, [0, 1, 2])  # hard fail: no draft policy
