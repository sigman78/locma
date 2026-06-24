from locma.core.actions import Attack, Pass, Summon, Use
from locma.core.views import BattleView, CardView
from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy


def test_random_draft_in_range():
    p = RandomPolicy("r", seed=1)
    assert p.draft_action(None, [0, 1, 2]) in (0, 1, 2)


def test_random_battle_returns_legal():
    p = RandomPolicy("r", seed=1)
    legal = [Pass()]
    assert p.battle_action(None, legal) in legal


# --- scripted: random draft ------------------------------------------------


def test_scripted_draft_in_range():
    picks = {ScriptedPolicy("s", seed=k).draft_action(None, [0, 1, 2]) for k in range(20)}
    assert picks <= {0, 1, 2}
    assert len(picks) > 1  # actually random, not a fixed pick


# --- scripted battle: priority order ---------------------------------------
# Order: green items on own creatures -> attack face (else clear Guard)
#        -> summon creatures -> use remaining items -> pass.


def _card(iid, type_):
    return CardView(iid, iid, type_, 1, 1, 1, "------")


def _view(*hand):
    return BattleView(1, 30, 5, 30, 0, tuple(hand), (), ())


def test_scripted_uses_green_items_first():
    view = _view(_card(10, 1))  # instance 10 is a green item
    legal = [Pass(), Summon(2), Attack(3, -1), Use(10, 99)]
    assert ScriptedPolicy("s").battle_action(view, legal) == Use(10, 99)


def test_scripted_attacks_face_before_summon_and_other_items():
    view = _view(_card(10, 2))  # red item in hand
    legal = [Pass(), Summon(2), Use(10, 5), Attack(3, -1)]
    assert ScriptedPolicy("s").battle_action(view, legal) == Attack(3, -1)


def test_scripted_clears_guard_when_face_not_legal():
    view = _view()
    legal = [Pass(), Summon(2), Attack(3, 7)]  # only a guard target is legal
    assert ScriptedPolicy("s").battle_action(view, legal) == Attack(3, 7)


def test_scripted_summons_before_remaining_items():
    view = _view(_card(10, 2))  # red item
    legal = [Pass(), Use(10, 5), Summon(2)]
    assert ScriptedPolicy("s").battle_action(view, legal) == Summon(2)


def test_scripted_uses_remaining_items_last():
    view = _view(_card(10, 3))  # blue item
    legal = [Pass(), Use(10, -1)]
    assert ScriptedPolicy("s").battle_action(view, legal) == Use(10, -1)


def test_scripted_battle_all_pass_returns_pass():
    result = ScriptedPolicy("s").battle_action(_view(), [Pass()])
    assert isinstance(result, Pass)


def test_scripted_targets_chosen_at_random():
    view = _view(_card(10, 1))  # green item, several legal targets
    uses = [Use(10, 1), Use(10, 2), Use(10, 3)]
    legal = [Pass(), *uses]
    chosen = {ScriptedPolicy("s", seed=k).battle_action(view, legal) for k in range(20)}
    assert chosen <= set(uses)
    assert len(chosen) > 1  # target actually varies
