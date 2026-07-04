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
    shared: bool = False,
) -> None:
    """Deal the draft pool and enter the DRAFT phase.

    Two rule variants:
      - default (LOCM): both seats pick independently from the same triplet — a
        pick does NOT deplete the offer, so both may take the same card.
      - ``shared``: a pick REMOVES the card from the other seat's offer (the
        second picker chooses from the remaining 2; the third card is burned).
        The first picker alternates each round (round r: seat ``r % 2``) for
        fairness — over 30 rounds each seat gets 15 first picks.
    """
    src = source if source is not None else ShuffledPoolSource()
    gs.draft_pool = src.build(cards, gs.rng, rounds)
    gs.draft_round = 0
    gs.current = 0
    gs.draft_shared = shared
    gs.draft_taken = None
    gs.phase = Phase.DRAFT


def draft_legal(gs: GameState) -> list[int]:
    if gs.draft_shared and gs.draft_taken is not None:
        return [i for i in range(3) if i != gs.draft_taken]
    return [0, 1, 2]


def current_triplet(gs: GameState) -> list[Card]:
    return gs.draft_pool[gs.draft_round]


def apply_draft_pick(gs: GameState, pick: int) -> None:
    if gs.draft_shared:
        _apply_shared_pick(gs, pick)
        return
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


def _apply_shared_pick(gs: GameState, pick: int) -> None:
    # Illegal picks are newly possible in this variant (the taken card), so
    # validate — the default variant accepts any of the 3 by construction.
    if pick not in draft_legal(gs):
        raise ValueError(
            f"illegal shared-draft pick {pick} "
            f"(round {gs.draft_round}: index {gs.draft_taken} already taken)"
        )
    player = gs.current
    gs.picks[player].append(gs.draft_pool[gs.draft_round][pick])
    if gs.draft_taken is None:  # first pick of the round
        gs.draft_taken = pick
        gs.current = gs.opponent(player)
    else:  # second pick closes the round
        gs.draft_taken = None
        gs.draft_round += 1
        gs.current = gs.draft_round % 2  # alternate who picks first
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
