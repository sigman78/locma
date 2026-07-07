from dataclasses import dataclass

import pytest

from locma.policies.drafts import (
    BalancedDraftPolicy,
    GreedyDraftPolicy,
    MaxAttackDraftPolicy,
    MaxDefenseDraftPolicy,
    MaxGuardDraftPolicy,
    PartialRandomDraftPolicy,
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


def test_balanced_item_discount_parameter():
    # Premium removal vs a modest creature: the default discount (12) rejects
    # the item, discount 0 takes it. Name encodes non-default discounts.
    offered = (_CV(2, 5, 0, -99, "BCDGLW"), _CV(0, 5, 4, 4, "------"))  # Decimate-like vs 4/4
    assert BalancedDraftPolicy().draft_action(_DV(offered), [0, 1]) == 1
    assert BalancedDraftPolicy(item_discount=0).draft_action(_DV(offered), [0, 1]) == 0
    assert BalancedDraftPolicy().name == "balanced-draft"
    assert BalancedDraftPolicy(item_discount=3).name == "balanced-draft-d3"


def test_registry_item_discount_specs():
    from locma.policies.registry import make_policy  # noqa: PLC0415

    # ppo: optional 2nd param sets the balanced item discount (model not loaded).
    p = make_policy("ppo:model.zip,3")
    assert p.draft.item_discount == 3.0
    assert make_policy("ppo:model.zip").draft.item_discount == 12.0
    # vbeam: optional 4th param, ensemble path syntax preserved.
    v = make_policy("vbeam:a.zip|b.zip,8,20,1.5")
    assert v.draft.item_discount == 1.5
    assert make_policy("vbeam:a.zip").draft.item_discount == 12.0


# ---------------------------------------------------------------------------
# PartialRandomDraftPolicy — k uniformly random picks on top of a base draft
# ---------------------------------------------------------------------------


@dataclass
class _RoundDV:
    round: int
    offered: tuple


class _ConstDraft:
    """Spy base: always picks 0 and counts how often it is consulted."""

    name = "const"

    def __init__(self):
        self.calls = 0

    def draft_action(self, view, legal):
        self.calls += 1
        return 0

    def reset(self, seed=None):
        self.calls = 0


def _round_view(r):
    return _RoundDV(r, (_CV(0, 1, 1, 1, "------"),) * 3)


def test_partial_random_overrides_exactly_k_rounds():
    base = _ConstDraft()
    p = PartialRandomDraftPolicy(base, k=4, seed=7)
    p.reset(7)
    for r in range(30):
        assert p.draft_action(_round_view(r), [0, 1, 2]) in (0, 1, 2)
    assert base.calls == 26  # exactly 30 - k delegated


def test_partial_random_both_seats_get_k_random_picks_each():
    # BattleEnv: the same policy drafts BOTH seats, alternating on each round.
    # Keying on view.round (not call count) gives each deck exactly k random picks.
    base = _ConstDraft()
    p = PartialRandomDraftPolicy(base, k=4, seed=7)
    p.reset(7)
    for r in range(30):
        p.draft_action(_round_view(r), [0, 1, 2])  # seat 0
        p.draft_action(_round_view(r), [0, 1, 2])  # seat 1
    assert base.calls == 52  # 2 x (30 - k)


def test_partial_random_reset_is_reproducible():
    p = PartialRandomDraftPolicy(_ConstDraft(), k=6, seed=0)
    p.reset(42)
    rounds = p._random_rounds
    first = [p.draft_action(_round_view(r), [0, 1, 2]) for r in range(30)]
    p.reset(42)
    assert p._random_rounds == rounds  # random rounds derive from the seed only
    assert [p.draft_action(_round_view(r), [0, 1, 2]) for r in range(30)] == first


def test_partial_random_k0_is_pure_delegation_and_k_bounds():
    base = _ConstDraft()
    p = PartialRandomDraftPolicy(base, k=0, seed=1)
    for r in range(30):
        assert p.draft_action(_round_view(r), [0, 1, 2]) == 0
    assert base.calls == 30
    with pytest.raises(ValueError):
        PartialRandomDraftPolicy(_ConstDraft(), k=31)
    with pytest.raises(ValueError):
        PartialRandomDraftPolicy(_ConstDraft(), k=-1)


def test_partial_random_name_and_stateful_base_tracking():
    base = BalancedDraftPolicy()
    p = PartialRandomDraftPolicy(base, k=30, seed=3)  # every round random
    assert p.name == "balanced-draft+rnd30"
    p.reset(3)
    for r in range(5):
        p.draft_action(_round_view(r), [0, 1, 2])
    # note_pick keeps the balanced draft's curve tracking accurate on overridden rounds
    assert len(base._picks) == 5
