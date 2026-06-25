from __future__ import annotations

from locma.core.actions import Attack, Pass, Summon
from locma.core.views import CardView, DraftView
from locma.policies.battles import GroundBattlePolicy
from locma.policies.drafts import MaxAttackDraftPolicy, MaxGuardDraftPolicy
from locma.policies.registry import make_policy

# --- helpers ---------------------------------------------------------------

GUARD = "---G--"  # mask over ABILITY_ORDER "BCDGLW"
NONE = "------"


def _creature(iid, attack=1, defense=1, abilities=NONE, cost=1):
    return CardView(iid, iid, 0, cost, attack, defense, abilities)


def _item(iid, attack=5, defense=0, cost=1):
    return CardView(iid, iid, 1, cost, attack, defense, NONE)


# --- GroundBattlePolicy ----------------------------------------------------


def test_ground_battle_prefers_face_attack():
    p = GroundBattlePolicy("g")
    legal = [Pass(), Summon(1), Attack(2, 5), Attack(2, -1)]
    assert p.battle_action(None, legal) == Attack(2, -1)


def test_ground_battle_clears_guard_when_face_not_legal():
    # When the enemy has a Guard, battle_legal offers no face attack.
    p = GroundBattlePolicy("g")
    legal = [Pass(), Summon(1), Attack(2, 5)]
    assert p.battle_action(None, legal) == Attack(2, 5)


def test_ground_battle_summons_when_no_attacks():
    p = GroundBattlePolicy("g")
    legal = [Pass(), Summon(7)]
    assert p.battle_action(None, legal) == Summon(7)


def test_ground_battle_passes_when_nothing_else():
    p = GroundBattlePolicy("g")
    assert isinstance(p.battle_action(None, [Pass()]), Pass)


# --- MaxGuardDraftPolicy ---------------------------------------------------


def test_max_guard_draft_picks_guard_creature():
    offered = (_creature(0, 4, 4), _creature(1, 1, 1, GUARD), _creature(2, 6, 6))
    view = DraftView(0, offered)
    assert MaxGuardDraftPolicy("mg").draft_action(view, [0, 1, 2]) == 1


def test_max_guard_draft_falls_back_to_best_creature_without_guard():
    offered = (_creature(0, 2, 2), _item(1), _creature(2, 5, 3))
    view = DraftView(0, offered)
    assert MaxGuardDraftPolicy("mg").draft_action(view, [0, 1, 2]) == 2


# --- MaxAttackDraftPolicy --------------------------------------------------


def test_max_attack_draft_picks_highest_attack_creature():
    offered = (_creature(0, 3, 9), _creature(1, 7, 1), _creature(2, 5, 5))
    view = DraftView(0, offered)
    assert MaxAttackDraftPolicy("ma").draft_action(view, [0, 1, 2]) == 1


def test_max_attack_draft_prefers_creature_over_higher_attack_item():
    offered = (_item(0, attack=9), _creature(1, 4, 4), _item(2, attack=8))
    view = DraftView(0, offered)
    assert MaxAttackDraftPolicy("ma").draft_action(view, [0, 1, 2]) == 1


def test_max_attack_uses_ground_battle():
    # In the split architecture, battle behaviour lives in the Composer's battle half.
    # The max-attack preset pairs MaxAttackDraftPolicy with GroundBattlePolicy.
    p = make_policy("max-attack")
    legal = [Pass(), Summon(1), Attack(2, -1)]
    assert p.battle_action(None, legal) == Attack(2, -1)


# --- registry --------------------------------------------------------------


def test_registry_resolves_baselines():
    assert make_policy("max-guard").name == "max-guard"
    assert make_policy("max-attack").name == "max-attack"
    assert isinstance(make_policy("max-guard").draft, MaxGuardDraftPolicy)
    assert isinstance(make_policy("max-attack").draft, MaxAttackDraftPolicy)
