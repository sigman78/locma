from __future__ import annotations

import pytest

from locma.data.cards_db import load_cards
from locma.harness.deck_quality import (
    card_cost_estimates,
    card_value,
    draft_deck,
    summarize_decks,
)
from locma.policies.drafts import BalancedDraftPolicy, RandomDraftPolicy, make_draft_policy


def test_card_value_uses_full_card_effects():
    cards = {c.id: c for c in load_cards()}
    decimate = cards[151]
    assert card_value(decimate) > card_value(cards[1])


def test_card_cost_estimates_have_cost_delta_for_all_cards():
    estimates = card_cost_estimates()
    assert len(estimates) == 160
    assert all(e.effective_cost == e.effective_cost for e in estimates)
    assert max(e.delta for e in estimates) > 0.0
    assert min(e.delta for e in estimates) < 0.0


def test_make_draft_policy_supports_draft_specs():
    assert make_draft_policy("draft:balanced").name == "balanced-draft"
    with pytest.raises(ValueError):
        make_draft_policy("draft:missing")


def test_draft_deck_is_deterministic_and_has_30_cards():
    p = RandomDraftPolicy(seed=7)
    deck_a = draft_deck(p, seed=11)
    deck_b = draft_deck(p, seed=11)
    assert [c.id for c in deck_a] == [c.id for c in deck_b]
    assert len(deck_a) == 30


def test_balanced_summary_tracks_curve_and_creatures():
    decks = [draft_deck(BalancedDraftPolicy(), seed=i) for i in range(3)]
    summary = summarize_decks("balanced", decks)
    assert summary.drafts == 3
    assert summary.avg_creatures > summary.avg_items
    assert summary.curve_l1 >= 0.0
