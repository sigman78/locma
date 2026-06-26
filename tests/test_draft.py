import random
from collections import Counter

from locma.core.draft import apply_draft_pick, current_triplet, draft_legal, start_draft
from locma.core.draft_source import ShuffledPoolSource
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards


def test_draft_runs_30_rounds_and_builds_decks():
    gs = GameState.new(random.Random(42))
    start_draft(gs, load_cards(), rounds=30)
    assert len(gs.draft_pool) == 30
    assert all(len(t) == 3 for t in gs.draft_pool)
    for _ in range(60):  # 30 rounds * 2 players
        assert draft_legal(gs) == [0, 1, 2]
        assert len(current_triplet(gs)) == 3
        apply_draft_pick(gs, 0)
    assert gs.phase == Phase.BATTLE
    assert len(gs.picks[0]) == 30 and len(gs.picks[1]) == 30
    assert len(gs.players[0].deck) == 30 and len(gs.players[1].deck) == 30


def test_draft_is_deterministic():
    a = GameState.new(random.Random(7))
    start_draft(a, load_cards())
    b = GameState.new(random.Random(7))
    start_draft(b, load_cards())
    assert [[c.id for c in t] for t in a.draft_pool] == [[c.id for c in t] for t in b.draft_pool]


def test_default_source_bounds_duplicates():
    # the default draft source is ShuffledPoolSource(copies=2)
    gs = GameState.new(random.Random(3))
    start_draft(gs, load_cards())
    counts = Counter(c.id for t in gs.draft_pool for c in t)
    assert max(counts.values()) <= 2


def test_start_draft_accepts_custom_source():
    gs = GameState.new(random.Random(3))
    start_draft(gs, load_cards(), source=ShuffledPoolSource(copies=1))
    ids = [c.id for t in gs.draft_pool for c in t]
    assert len(ids) == len(set(ids))  # copies=1 deals a single shuffled deck -> distinct
