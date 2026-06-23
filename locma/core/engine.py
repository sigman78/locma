from __future__ import annotations
import random
from dataclasses import dataclass

from locma.core.state import GameState, Phase
from locma.core import draft as draftmod
from locma.core import battle as battlemod
from locma.core.views import DraftView, BattleView, CardView
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
    )


def make_draft_view(gs: GameState) -> DraftView:
    """Build a sanitized DraftView from the current game state."""
    offered = tuple(
        CardView(-1, c.id, int(c.type), c.cost, c.attack, c.defense, c.abilities)
        for c in gs.draft_pool[gs.draft_round]
    )
    return DraftView(gs.draft_round, offered)


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


def run_game(policy0, policy1, seed: int, cards=None, max_turns: int = 200) -> GameResult:
    """Drive a complete LOCM 1.2 game (draft then battle) between two policies.

    Determinism guarantee: the same seed + same policies always produce the
    same (winner, turns).  The game's RNG is seeded exclusively via
    random.Random(seed); policy RNGs are independent and not touched here.

    Turn-change detection: the battle inner loop tracks `turn_owner = gs.current`
    before any action is applied.  After each apply_battle call we check
    `gs.current != turn_owner` (Pass/end_turn changes gs.current) to detect
    that the turn has ended and break out of the inner loop.

    Safety caps:
      - per-turn action cap of 100 actions forces end_turn to prevent infinite loops.
      - global safety cap of 1000 iterations (shared across all half-turns).
      - if gs.turn > max_turns and no winner yet, winner = player with higher
        health (tiebreak → player 0).
    """
    cards = cards or load_cards()
    gs = GameState.new(random.Random(seed))

    # --- Draft phase ---
    draftmod.start_draft(gs, cards)
    pols = (policy0, policy1)
    while gs.phase == Phase.DRAFT:
        view = make_draft_view(gs)
        pick = pols[gs.current].draft_action(view, [0, 1, 2])
        draftmod.apply_draft_pick(gs, pick)

    # --- Battle phase ---
    battlemod.start_battle(gs)
    safety = 0
    while gs.phase == Phase.BATTLE and gs.turn <= max_turns:
        per_turn = 0
        turn_owner = gs.current
        # Inner loop: keep taking actions until the turn changes or game ends
        while gs.current == turn_owner and gs.phase == Phase.BATTLE:
            legal = battlemod.battle_legal(gs)
            view = make_battle_view(gs)
            action = pols[gs.current].battle_action(view, legal)
            battlemod.apply_battle(gs, action)
            per_turn += 1
            if per_turn > 100:
                # Safety: force end-of-turn to prevent policy from looping forever
                battlemod.end_turn(gs)
                break
        safety += 1
        if safety > 1000:
            break

    # --- Resolve winner if game did not end normally ---
    if gs.winner is None:
        h0, h1 = gs.players[0].health, gs.players[1].health
        gs.winner = 0 if h0 >= h1 else 1

    return GameResult(winner=gs.winner, turns=gs.turn, seed=seed)
