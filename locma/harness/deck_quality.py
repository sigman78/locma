from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

from locma.core import draft as draftmod
from locma.core.cards import ABILITY_ORDER, Card, CardType
from locma.core.engine import make_draft_view
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards

_CREATURE = CardType.CREATURE
_ABILITY_WEIGHTS = {
    "B": 0.8,
    "C": 1.2,
    "D": 1.1,
    "G": 2.0,
    "L": 1.8,
    "W": 1.7,
}
_CURVE_TARGET = {0: 1, 1: 3, 2: 5, 3: 5, 4: 5, 5: 4, 6: 3, 7: 4}
_REMOVAL_CAP = 13.0


@dataclass(frozen=True)
class CardCostEstimate:
    card_id: int
    name: str
    type: str
    cost: int
    value: float
    effective_cost: float
    delta: float


@dataclass(frozen=True)
class DeckSummary:
    policy: str
    drafts: int
    quality: float
    avg_card_value: float
    avg_effective_cost_delta: float
    avg_cost: float
    avg_creatures: float
    avg_items: float
    curve_l1: float
    avg_draw: float
    avg_guards: float
    avg_wards: float
    avg_lethals: float


def card_value(card: Card) -> float:
    """Draft-quality proxy using full card text fields.

    This is intentionally transparent and cheap, not a learned oracle. Positive
    surplus means the card carries more stats/effects than its printed cost would
    suggest under this proxy; it is a ranking feature for experiment selection.
    """
    abilities = sum(
        _ABILITY_WEIGHTS[ch] for i, ch in enumerate(ABILITY_ORDER) if card.abilities[i] != "-"
    )
    hp_swing = 0.8 * card.player_hp - 1.0 * card.enemy_hp
    draw = 2.2 * card.card_draw
    if card.type == _CREATURE:
        return max(0.0, card.attack) + max(0.0, card.defense) + abilities + hp_swing + draw
    stat_effect = min(abs(card.attack), _REMOVAL_CAP) + min(abs(card.defense), _REMOVAL_CAP)
    if card.abilities == ABILITY_ORDER:
        stat_effect += 2.5
    return stat_effect + abilities + hp_swing + draw


def card_cost_estimates(cards: list[Card] | None = None) -> list[CardCostEstimate]:
    cards = cards or load_cards()
    values = np.asarray([card_value(c) for c in cards], dtype=float)
    costs = np.asarray([c.cost for c in cards], dtype=float)
    slope, intercept = np.polyfit(values, costs, deg=1)
    estimates: list[CardCostEstimate] = []
    for card, value in zip(cards, values, strict=True):
        effective = float(slope * value + intercept)
        estimates.append(
            CardCostEstimate(
                card_id=card.id,
                name=card.name,
                type=card.type.name.lower(),
                cost=card.cost,
                value=float(value),
                effective_cost=effective,
                delta=effective - card.cost,
            )
        )
    return estimates


def draft_deck(policy, seed: int, cards: list[Card] | None = None) -> list[Card]:
    """Return one 30-card deck drafted by ``policy`` from the default pool source."""
    cards = cards or load_cards()
    policy.reset(seed)
    gs = GameState.new(random.Random(seed))
    draftmod.start_draft(gs, cards)
    while gs.phase == Phase.DRAFT:
        if gs.current == 0:
            view = make_draft_view(gs)
            pick = policy.draft_action(view, draftmod.draft_legal(gs))
            draftmod.apply_draft_pick(gs, pick)
        else:
            # Seat 1 is irrelevant for this one-policy deck probe; advance with
            # a deterministic neutral pick so the shared pool/rounds progress.
            draftmod.apply_draft_pick(gs, 0)
    return list(gs.picks[0])


def summarize_decks(policy_name: str, decks: list[list[Card]]) -> DeckSummary:
    if not decks:
        raise ValueError("at least one deck is required")
    estimates = {e.card_id: e for e in card_cost_estimates()}
    qualities: list[float] = []
    avg_values: list[float] = []
    deltas: list[float] = []
    costs: list[float] = []
    creatures: list[int] = []
    items: list[int] = []
    curve_l1s: list[int] = []
    draws: list[int] = []
    guards: list[int] = []
    wards: list[int] = []
    lethals: list[int] = []
    for deck in decks:
        vals = [card_value(c) for c in deck]
        avg_values.append(float(np.mean(vals)))
        deltas.append(float(np.mean([estimates[c.id].delta for c in deck])))
        costs.append(float(np.mean([c.cost for c in deck])))
        creature_count = sum(1 for c in deck if c.type == _CREATURE)
        creatures.append(creature_count)
        items.append(len(deck) - creature_count)
        buckets = Counter(min(c.cost, 7) for c in deck)
        curve_l1 = sum(abs(buckets.get(k, 0) - v) for k, v in _CURVE_TARGET.items())
        curve_l1s.append(curve_l1)
        draws.append(sum(c.card_draw for c in deck))
        guards.append(sum(c.has("G") for c in deck if c.type == _CREATURE))
        wards.append(sum(c.has("W") for c in deck if c.type == _CREATURE))
        lethals.append(sum(c.has("L") for c in deck if c.type == _CREATURE))
        qualities.append(
            float(np.mean(vals) + np.mean([estimates[c.id].delta for c in deck]) - 0.15 * curve_l1)
        )
    return DeckSummary(
        policy=policy_name,
        drafts=len(decks),
        quality=float(np.mean(qualities)),
        avg_card_value=float(np.mean(avg_values)),
        avg_effective_cost_delta=float(np.mean(deltas)),
        avg_cost=float(np.mean(costs)),
        avg_creatures=float(np.mean(creatures)),
        avg_items=float(np.mean(items)),
        curve_l1=float(np.mean(curve_l1s)),
        avg_draw=float(np.mean(draws)),
        avg_guards=float(np.mean(guards)),
        avg_wards=float(np.mean(wards)),
        avg_lethals=float(np.mean(lethals)),
    )


def summarize_policy(policy, policy_name: str, drafts: int = 200, seed: int = 0) -> DeckSummary:
    if drafts < 1:
        raise ValueError("drafts must be >= 1")
    cards = load_cards()
    decks = [draft_deck(policy, seed + i, cards) for i in range(drafts)]
    return summarize_decks(policy_name, decks)
