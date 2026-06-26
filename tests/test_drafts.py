from dataclasses import dataclass

from locma.policies.drafts import (
    BalancedDraftPolicy,
    GreedyDraftPolicy,
    MaxAttackDraftPolicy,
    MaxDefenseDraftPolicy,
    MaxGuardDraftPolicy,
    RandomDraftPolicy,
    WeightedDraftPolicy,
)


@dataclass
class _CV:
    type: int
    cost: int
    attack: int
    defense: int
    abilities: str


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
