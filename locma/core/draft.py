from __future__ import annotations

from locma.core.cards import Card
from locma.core.draft_source import DraftSource, ShuffledPoolSource
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase


def start_draft(
    gs: GameState,
    cards: list[Card],
    rounds: int = 30,
    source: DraftSource | None = None,
) -> None:
    src = source if source is not None else ShuffledPoolSource()
    gs.draft_pool = src.build(cards, gs.rng, rounds)
    gs.draft_round = 0
    gs.current = 0
    gs.phase = Phase.DRAFT


def draft_legal(gs: GameState) -> list[int]:
    return [0, 1, 2]


def current_triplet(gs: GameState) -> list[Card]:
    return gs.draft_pool[gs.draft_round]


def apply_draft_pick(gs: GameState, pick: int) -> None:
    player = gs.current
    chosen = gs.draft_pool[gs.draft_round][pick]
    gs.picks[player].append(chosen)
    if gs.current == 0:
        gs.current = 1
    else:
        gs.current = 0
        gs.draft_round += 1
        if gs.draft_round >= len(gs.draft_pool):
            _finish_draft(gs)


def _finish_draft(gs: GameState) -> None:
    iid = 0
    for p in (0, 1):
        deck = []
        for card in gs.picks[p]:
            deck.append(CardInstance.from_card(card, iid))
            iid += 1
        gs.rng.shuffle(deck)
        gs.players[p].deck = deck
    gs.phase = Phase.BATTLE
    gs.draft_round = 0
    gs.current = 0
