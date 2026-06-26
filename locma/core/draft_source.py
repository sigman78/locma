"""Draft pool sources.

A :class:`DraftSource` builds the shared draft pool — ``rounds`` triplets of
``per_round`` cards that both seats draft from, round by round. The pool is the
only randomness the draft phase consumes, so a source must draw exclusively
from the supplied ``rng`` to keep seeded games reproducible.
"""

from __future__ import annotations

import random
from typing import Protocol

from locma.core.cards import Card


class DraftSource(Protocol):
    def build(
        self, cards: list[Card], rng: random.Random, rounds: int, per_round: int = 3
    ) -> list[list[Card]]:
        """Return ``rounds`` triplets of ``per_round`` cards, shared by both seats."""
        ...


class ShuffledPoolSource:
    """Deal the pool from the whole card space duplicated ``copies`` times.

    Builds ``cards * copies``, shuffles it, and deals the offered cards
    sequentially. Every card therefore appears at most ``copies`` times across
    the whole pool — an even spread with bounded duplicates. (Two copies of one
    card can still land in the same triplet.) This is the default draft source.
    """

    def __init__(self, copies: int = 2) -> None:
        self.copies = copies

    def build(
        self, cards: list[Card], rng: random.Random, rounds: int, per_round: int = 3
    ) -> list[list[Card]]:
        needed = rounds * per_round
        available = len(cards) * self.copies
        if available < needed:
            raise ValueError(
                f"draft pool too small: {len(cards)} cards * {self.copies} copies "
                f"= {available} < {needed} needed ({rounds} rounds * {per_round})"
            )
        deck = cards * self.copies
        rng.shuffle(deck)
        return [deck[i * per_round : (i + 1) * per_round] for i in range(rounds)]


class UniformSource:
    """Sample each offered card independently and uniformly, with replacement.

    The original draft distribution: every slot is an independent uniform draw,
    so a card may appear any number of times across the pool.
    """

    def build(
        self, cards: list[Card], rng: random.Random, rounds: int, per_round: int = 3
    ) -> list[list[Card]]:
        return [
            [cards[rng.randint(0, len(cards) - 1)] for _ in range(per_round)] for _ in range(rounds)
        ]
