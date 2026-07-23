from __future__ import annotations

import random
from dataclasses import dataclass

from locma.core import battle as battlemod
from locma.core import draft as draftmod
from locma.core.state import GameState, Phase
from locma.core.views import BattleView, CardView, DraftView
from locma.data.cards_db import load_cards


@dataclass(frozen=True)
class GameResult:
    winner: int
    turns: int
    seed: int


def _cv(inst, hide_id: bool = False) -> CardView:
    """Convert a CardInstance to a CardView, optionally hiding instance_id."""
    return CardView(
        instance_id=inst.instance_id if not hide_id else -1,
        card_id=inst.card.id,
        type=int(inst.card.type),
        cost=inst.card.cost,
        attack=inst.attack,
        defense=inst.defense,
        abilities=inst.abilities,
        can_attack=inst.can_attack,
        has_attacked=inst.has_attacked,
    )


def make_draft_view(gs: GameState) -> DraftView:
    """Build a sanitized DraftView from the current game state."""
    offered = tuple(
        CardView(-1, c.id, int(c.type), c.cost, c.attack, c.defense, c.abilities)
        for c in gs.draft_pool[gs.draft_round]
    )
    return DraftView(gs.draft_round, offered, gs.draft_taken)


def make_battle_view(gs: GameState) -> BattleView:
    """Build a sanitized BattleView for the current player.

    Opponent hand contents are never exposed — only the count is included.
    """
    me = gs.players[gs.current]
    op = gs.players[gs.opponent(gs.current)]
    return BattleView(
        turn=gs.turn,
        me_health=me.health,
        me_mana=me.mana,
        op_health=op.health,
        op_hand_count=len(op.hand),  # count only — contents never exposed
        my_hand=tuple(_cv(c) for c in me.hand),
        my_board=tuple(_cv(c) for c in me.board),
        op_board=tuple(_cv(c) for c in op.board),
    )


def run_game(
    policy0,
    policy1,
    seed: int,
    cards=None,
    max_turns: int = 200,
    on_step=None,
    on_snapshot=None,
    on_pre_step=None,
    on_event=None,
    shared_draft: bool = False,
) -> GameResult:
    """Drive a complete LOCM 1.2 game (draft then battle) between two policies.

    ``shared_draft`` switches the draft to the shared variant — a pick removes
    the card from the other seat's offer and the first picker alternates by
    round (see ``draft.start_draft``). Default is the LOCM rule: both seats
    pick independently from the same triplet.

    Determinism guarantee: the same seed + same policies always produce the
    same (winner, turns).  The game's RNG is seeded exclusively via
    random.Random(seed); policy RNGs are reset to the same seed so that in a
    mirrored pair run_game(A,B,s) and run_game(B,A,s) policy A sees identical
    randomness regardless of seat (clean mirror control).

    Turn-change detection: the battle inner loop tracks `turn_owner = gs.current`
    before any action is applied.  After each apply_battle call we check
    `gs.current != turn_owner` (Pass/end_turn changes gs.current) to detect
    that the turn has ended and break out of the inner loop.

    Recording hooks (all optional):
      - on_snapshot(gs): fired once at battle start (the opening state).
      - on_pre_step(seat, action, gs): fired with the decision-point state, just
        BEFORE each battle action is applied.  This is the actor's own
        perspective (gs.current == seat) — the natural state to record, since a
        Pass's apply_battle runs end_turn() and the post-apply state belongs to
        the opponent.
      - on_event(ev): fired for each atomic engine event (damage, unit_died,
        turn_ended, turn_started). None = no emission.
      - on_step(seat, action, gs): fired AFTER each draft/battle action.

    Safety caps:
      - per-turn action cap of 100 actions forces end_turn to prevent infinite loops.
      - global safety cap of 1000 iterations (shared across all half-turns).
      - if gs.turn > max_turns and no winner yet, winner = player with higher
        health (tiebreak → player 0).
    """
    cards = cards or load_cards()
    # Reset policies so each game's randomness is a deterministic function of
    # the game seed only — independent of how many prior games were played.
    policy0.reset(seed)
    policy1.reset(seed)
    gs = GameState.new(random.Random(seed))

    # --- Draft phase ---
    draftmod.start_draft(gs, cards, shared=shared_draft)
    pols = (policy0, policy1)
    while gs.phase == Phase.DRAFT:
        seat = gs.current
        view = make_draft_view(gs)
        pick = pols[gs.current].draft_action(view, draftmod.draft_legal(gs))
        draftmod.apply_draft_pick(gs, pick)
        if on_step is not None:
            on_step(seat, pick, gs)

    # --- Battle phase ---
    return _run_battle(gs, pols, seed, max_turns, on_snapshot, on_pre_step, on_step, on_event)


def _run_battle(
    gs, pols, seed, max_turns, on_snapshot, on_pre_step, on_step, on_event
) -> GameResult:
    """Drive the battle phase to completion. Shared by ``run_game`` (after its
    draft) and ``run_battle_from_decks`` (which injects pre-drafted decks). ``gs``
    must already have both ``players[*].deck`` populated and ``phase == BATTLE``."""
    battlemod.start_battle(gs, emit=on_event)
    if on_snapshot is not None:
        on_snapshot(gs)
    safety = 0
    while gs.phase == Phase.BATTLE and gs.turn <= max_turns:
        per_turn = 0
        turn_owner = gs.current
        # Inner loop: keep taking actions until the turn changes or game ends
        while gs.current == turn_owner and gs.phase == Phase.BATTLE:
            seat = gs.current
            legal = battlemod.battle_legal(gs)
            view = make_battle_view(gs)
            action = pols[gs.current].battle_action(view, legal, gs)
            if on_pre_step is not None:
                # Decision point: the game state as the acting seat sees it, BEFORE
                # apply_battle runs (a turn-ending Pass calls end_turn here, which
                # flips gs.current and draws for the opponent — so post-apply state
                # no longer belongs to `seat`).
                on_pre_step(seat, action, gs)
            battlemod.apply_battle(gs, action, emit=on_event)
            if on_step is not None:
                on_step(seat, action, gs)
            per_turn += 1
            if per_turn > 100:
                battlemod.end_turn(gs, emit=on_event)
                break
        safety += 1
        if safety > 1000:
            break

    # --- Resolve winner if game did not end normally ---
    if gs.winner is None:
        h0, h1 = gs.players[0].health, gs.players[1].health
        gs.winner = 0 if h0 >= h1 else 1

    return GameResult(winner=gs.winner, turns=gs.turn, seed=seed)


def run_battle_from_decks(
    deck0: list[int],
    deck1: list[int],
    policy0,
    policy1,
    seed: int,
    cards=None,
    max_turns: int = 200,
    on_step=None,
    on_snapshot=None,
    on_pre_step=None,
    on_event=None,
) -> GameResult:
    """Play a battle from two PRE-DRAFTED decks, skipping the draft phase.

    ``deck0``/``deck1`` are lists of card ids (the 30 cards each seat drafted;
    order irrelevant — the deck is reshuffled by the game rng). This is the
    injection path for a cached deck pool: it reproduces exactly what
    ``draft._finish_draft`` builds (``CardInstance.from_card`` with a global
    instance-id counter, then ``gs.rng.shuffle``) so the deck objects are built
    just as a live draft would — only the ~60 draft policy calls are skipped.
    Determinism: same seed + decks + policies -> same (winner, turns) (fresh rng,
    no draft consumption), and the per-seed reshuffle keeps draw order varying so
    the same deck is not replayed as the same game. (Draw order differs from a
    live ``run_game`` at the same seed, since the draft consumes rng there —
    irrelevant for training/benching on cached decks.)
    """
    cards = cards or load_cards()
    policy0.reset(seed)
    policy1.reset(seed)
    gs = GameState.new(random.Random(seed))
    deal_decks(gs, deck0, deck1, cards)
    pols = (policy0, policy1)
    return _run_battle(gs, pols, seed, max_turns, on_snapshot, on_pre_step, on_step, on_event)


def deal_decks(gs, deck0: list[int], deck1: list[int], cards=None) -> None:
    """Populate ``gs.players[*].deck`` from two card-id lists and set phase=BATTLE.

    Reproduces ``draft._finish_draft`` (``CardInstance.from_card`` with a global
    instance-id counter, then ``gs.rng.shuffle``) so a cached deck is dealt
    exactly as a live draft would build it. Shared by ``run_battle_from_decks``
    and ``BattleEnv.reset`` (deck-pool path)."""
    from locma.core.instance import CardInstance  # noqa: PLC0415

    cards = cards or load_cards()
    by_id = {c.id: c for c in (cards if isinstance(cards, list) else cards.values())}
    iid = 0
    for p, deck_ids in ((0, deck0), (1, deck1)):
        deck = []
        for cid in deck_ids:
            deck.append(CardInstance.from_card(by_id[cid], iid))
            iid += 1
        gs.rng.shuffle(deck)
        gs.players[p].deck = deck
    gs.phase = Phase.BATTLE
