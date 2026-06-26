import random
from collections import Counter

import pytest

from locma.core.draft_source import ShuffledPoolSource, UniformSource
from locma.data.cards_db import load_cards

CARDS = load_cards()


def _ids(pool):
    return [c.id for t in pool for c in t]


def test_shuffled_pool_shape():
    pool = ShuffledPoolSource().build(CARDS, random.Random(0), rounds=30)
    assert len(pool) == 30
    assert all(len(t) == 3 for t in pool)


def test_shuffled_pool_deterministic():
    a = ShuffledPoolSource().build(CARDS, random.Random(7), 30)
    b = ShuffledPoolSource().build(CARDS, random.Random(7), 30)
    assert [[c.id for c in t] for t in a] == [[c.id for c in t] for t in b]


def test_shuffled_pool_default_copies_bounds_duplicates():
    # default copies=2 -> no card appears more than twice across the whole pool
    pool = ShuffledPoolSource(copies=2).build(CARDS, random.Random(3), 30)
    counts = Counter(_ids(pool))
    assert max(counts.values()) <= 2
    assert sum(counts.values()) == 90  # rounds * per_round


def test_shuffled_pool_copies_one_is_distinct():
    # copies=1 deals from a single shuffled deck -> every offered card is unique
    pool = ShuffledPoolSource(copies=1).build(CARDS, random.Random(1), 30)
    ids = _ids(pool)
    assert len(ids) == len(set(ids))


def test_shuffled_pool_raises_when_pool_too_small():
    small = CARDS[:5]  # 5 * 1 = 5 < 30 * 3 = 90
    with pytest.raises(ValueError):
        ShuffledPoolSource(copies=1).build(small, random.Random(0), rounds=30)


def test_shuffled_pool_respects_per_round():
    pool = ShuffledPoolSource().build(CARDS, random.Random(0), rounds=10, per_round=4)
    assert len(pool) == 10
    assert all(len(t) == 4 for t in pool)


def test_uniform_source_shape_and_deterministic():
    a = UniformSource().build(CARDS, random.Random(5), 30)
    b = UniformSource().build(CARDS, random.Random(5), 30)
    assert len(a) == 30 and all(len(t) == 3 for t in a)
    assert [[c.id for c in t] for t in a] == [[c.id for c in t] for t in b]
