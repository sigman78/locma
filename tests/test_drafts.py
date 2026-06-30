from dataclasses import dataclass

from locma.policies.drafts import (
    AggroDraftPolicy,
    BalancedDraftPolicy,
    DefenseDraftPolicy,
    GreedyDraftPolicy,
    MaxAttackDraftPolicy,
    MaxDefenseDraftPolicy,
    MaxGuardDraftPolicy,
    MidrangeDraftPolicy,
    RandomDraftPolicy,
    TrueCostBalancedDraftPolicy,
    WeightedBalancedDraftPolicy,
    WeightedDraftPolicy,
    make_draft_policy,
)


@dataclass
class _CV:
    type: int
    cost: int
    attack: int
    defense: int
    abilities: str
    card_id: int = 1


@dataclass
class _DV:
    offered: tuple


def _view():
    # idx0 plain creature, idx1 guard creature, idx2 big vanilla creature
    return _DV(
        offered=(
            _CV(0, 1, 1, 1, "------"),
            _CV(0, 2, 2, 2, "--G---"),
            _CV(0, 3, 5, 5, "------"),
        )
    )


def test_random_draft_reproducible_after_reset():
    p = RandomDraftPolicy("r", seed=1)
    first = [p.draft_action(None, [0, 1, 2]) for _ in range(5)]
    p.reset(1)
    assert [p.draft_action(None, [0, 1, 2]) for _ in range(5)] == first


def test_max_guard_prefers_guard():
    assert MaxGuardDraftPolicy("mg").draft_action(_view(), [0, 1, 2]) == 1


def test_max_attack_prefers_highest_attack():
    assert MaxAttackDraftPolicy("ma").draft_action(_view(), [0, 1, 2]) == 2


def test_greedy_draft_prefers_highest_score():
    assert GreedyDraftPolicy("g").draft_action(_view(), [0, 1, 2]) == 2


def test_max_defense_prefers_highest_defense_creature():
    offered = (_CV(0, 1, 5, 1, "------"), _CV(0, 2, 1, 5, "------"), _CV(1, 1, 9, 9, "------"))
    assert MaxDefenseDraftPolicy().draft_action(_DV(offered), [0, 1, 2]) == 1


def test_weighted_values_keywords_over_raw_stats():
    # idx0 vanilla 3/3 (no kw); idx1 guard 3/2 (G in proper BCDGLW slot 3).
    offered = (_CV(0, 3, 3, 3, "------"), _CV(0, 3, 3, 2, "---G--"), _CV(0, 1, 1, 1, "------"))
    # weighted values Guard (+2.0) so 3+2+2.0=7 beats the vanilla 6 ...
    assert WeightedDraftPolicy().draft_action(_DV(offered), [0, 1, 2]) == 1
    # ... whereas plain greedy (0.5/keyword) takes the higher-stat vanilla.
    assert GreedyDraftPolicy().draft_action(_DV(offered), [0, 1, 2]) == 0


def test_weighted_values_removal_spell_by_effect_magnitude():
    # A red item with -7 defense (7 damage to an ENEMY minion) must not be scored
    # as a -7 card; spell-aware value treats it as strong removal (+7).
    offered = (
        _CV(2, 7, 0, -7, "------"),  # red item: 7-damage removal
        _CV(0, 2, 1, 1, "------"),  # weak 1/1 creature
        _CV(0, 1, 1, 1, "------"),
    )
    assert WeightedDraftPolicy().draft_action(_DV(offered), [0, 1, 2]) == 0


def test_card_value_caps_destroy_sentinel():
    from locma.policies.drafts import _card_value  # noqa: PLC0415

    decimate = _CV(2, 5, 0, -99, "BCDGLW")  # destroy-a-minion + strip-all sentinel
    v = _card_value(decimate)
    assert v < 25, "defense -99 must be capped, not ~99"
    assert v > 6, "still premium: above a vanilla 3/3 creature (6)"


def test_balanced_prefers_creature_and_is_stateful():
    p = BalancedDraftPolicy()
    offered = (_CV(1, 3, 2, 2, "------"), _CV(0, 3, 2, 2, "------"))  # item vs equal creature
    assert p.draft_action(_DV(offered), [0, 1]) == 1  # creature preferred
    assert len(p._picks) == 1
    p.reset()
    assert p._picks == []


def test_weighted_balanced_admits_premium_removal_more_than_balanced():
    offered = (
        _CV(2, 5, 0, -13, "------", 110),
        _CV(0, 5, 3, 3, "------", 1),
        _CV(0, 1, 1, 1, "------", 2),
    )
    assert BalancedDraftPolicy().draft_action(_DV(offered), [0, 1, 2]) != 0
    assert WeightedBalancedDraftPolicy().draft_action(_DV(offered), [0, 1, 2]) == 0


def test_truecost_balanced_uses_card_id_for_draw_and_hp_effects():
    # Card 149 draws a card and strips all abilities; a same-view fake card id
    # lacks those full-card fields, so the true-cost policy should prefer 149.
    offered = (
        _CV(2, 3, 0, 0, "BCDGLW", 149),
        _CV(2, 0, 0, 0, "BCDGLW", 142),
        _CV(0, 1, 2, 1, "------", 1),
    )
    assert TrueCostBalancedDraftPolicy().draft_action(_DV(offered), [0, 1, 2]) == 0


def test_make_draft_policy_supports_new_research_drafts():
    assert isinstance(make_draft_policy("draft:weighted-balanced"), WeightedBalancedDraftPolicy)
    assert isinstance(make_draft_policy("truecost-balanced"), TrueCostBalancedDraftPolicy)


def test_aggro_prefers_cheap_attack_over_expensive_body():
    offered = (
        _CV(0, 2, 4, 1, "-C----", 39),
        _CV(0, 6, 4, 7, "------", 68),
        _CV(0, 3, 2, 3, "---G--", 10),
    )
    assert AggroDraftPolicy().draft_action(_DV(offered), [0, 1, 2]) == 0


def test_defense_prefers_guard_and_defense():
    offered = (
        _CV(0, 2, 4, 1, "-C----", 39),
        _CV(0, 3, 2, 5, "---G--", 96),
        _CV(0, 3, 5, 2, "------", 93),
    )
    assert DefenseDraftPolicy().draft_action(_DV(offered), [0, 1, 2]) == 1


def test_midrange_and_defense_factory():
    assert isinstance(make_draft_policy("draft:midrange"), MidrangeDraftPolicy)
    assert isinstance(make_draft_policy("draft:defense"), DefenseDraftPolicy)
