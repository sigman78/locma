from dataclasses import dataclass

from locma.policies.drafts import (
    GreedyDraftPolicy,
    MaxAttackDraftPolicy,
    MaxGuardDraftPolicy,
    RandomDraftPolicy,
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
