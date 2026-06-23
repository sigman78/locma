from __future__ import annotations
from locma.core.cards import Card
from locma.core.state import GameState, Phase
from locma.core.instance import CardInstance

def start_draft(gs: GameState, cards: list[Card], rounds: int = 30) -> None:
    pool = []
    for _ in range(rounds):
        triplet = [cards[gs.rng.randint(0, len(cards) - 1)] for _ in range(3)]
        pool.append(triplet)
    gs.draft_pool = pool
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
            deck.append(CardInstance.from_card(card, iid)); iid += 1
        gs.rng.shuffle(deck)
        gs.players[p].deck = deck
    gs.phase = Phase.BATTLE
    gs.draft_round = 0
    gs.current = 0
