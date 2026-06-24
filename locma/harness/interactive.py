# locma/harness/interactive.py
from __future__ import annotations

import random

from locma.core import battle as battlemod
from locma.core import draft as draftmod
from locma.core.actions import action_to_dict
from locma.core.engine import make_draft_view
from locma.core.state import GameState, Phase
from locma.harness.replay_stream import (
    StreamRecorder,
    _card_dict,
    _creature_dict,
)

MAX_TURNS = 200


class IllegalMove(Exception):
    """Submitted move is not legal in the current state."""


class WrongPhase(Exception):
    """Submitted move does not match the phase / seat to act."""


class InteractiveGame:
    """One human-vs-AI game, advanced on demand.

    The driver pauses whenever the current seat is the human's. AI seats resolve
    inline. It mirrors run_game's loop and reuses the same engine primitives and
    recording hooks so a finished game is a standard locma-replay/2.
    """

    def __init__(self, game_id, ai_policy, seed, human_seat, cards):
        self.game_id = game_id
        self.ai = ai_policy
        self.seed = seed
        self.human_seat = human_seat
        self.cards = cards
        self.gs = GameState.new(random.Random(seed))
        self.ai.reset(seed)
        self.rec = StreamRecorder()
        self._battle_started = False
        self._slice: list[dict] = []
        self.result: dict | None = None
        self._replay: dict | None = None
        draftmod.start_draft(self.gs, cards)

    # -- recording: feed the recorder AND the per-response slice buffer --
    def _emit(self, ev):
        self.rec.on_event(ev)
        self._slice.append(ev)

    def start(self):
        self._advance()
        return self

    @property
    def status(self) -> str:
        return "finished" if self.result is not None else "awaiting_human"

    # -- advance until the next human decision or game end --
    def _advance(self):
        gs = self.gs
        while gs.phase == Phase.DRAFT and gs.current != self.human_seat:
            seat = gs.current
            pick = self.ai.draft_action(make_draft_view(gs), [0, 1, 2])
            draftmod.apply_draft_pick(gs, pick)
            self.rec.on_step(seat, pick, gs)
        if gs.phase == Phase.DRAFT:
            return  # human's draft pick needed
        # (battle transition + battle loop added in Task 3)

    def submit_draft(self, pick: int) -> dict:
        gs = self.gs
        if self.result is not None or gs.phase != Phase.DRAFT or gs.current != self.human_seat:
            raise WrongPhase("not your draft turn")
        if pick not in (0, 1, 2):
            raise IllegalMove(f"bad pick: {pick!r}")
        self._slice = []
        seat = gs.current
        draftmod.apply_draft_pick(gs, pick)
        self.rec.on_step(seat, pick, gs)
        self._advance()
        return self._response()

    # -- payload builders --
    def _play_view(self) -> dict:
        gs = self.gs
        me = gs.players[self.human_seat]
        op = gs.players[gs.opponent(self.human_seat)]
        return {
            "turn": gs.turn,
            "me": {
                "health": me.health,
                "mana": me.mana,
                "max_mana": me.max_mana,
                "deck_count": len(me.deck),
                "bonus_draw": me.bonus_draw,
                "hand": [_card_dict(c) for c in me.hand],
                "board": [_creature_dict(c) for c in me.board],
            },
            "op": {
                "health": op.health,
                "mana": op.mana,
                "max_mana": op.max_mana,
                "deck_count": len(op.deck),
                "bonus_draw": op.bonus_draw,
                "hand_count": len(op.hand),
                "board": [_creature_dict(c) for c in op.board],
            },
        }

    def pending(self) -> dict | None:
        gs = self.gs
        if self.result is not None:
            return None
        if gs.phase == Phase.DRAFT:
            trip = draftmod.current_triplet(gs)
            return {
                "phase": "draft",
                "you": self.human_seat,
                "round": gs.draft_round,
                "total": len(gs.draft_pool),
                "triplet": [c.id for c in trip],
                "my_picks": len(gs.picks[self.human_seat]),
            }
        legal = battlemod.battle_legal(gs)
        return {
            "phase": "battle",
            "you": self.human_seat,
            "view": self._play_view(),
            "legal": [action_to_dict(a) for a in legal],
        }

    def _response(self) -> dict:
        return {
            "status": self.status,
            "slice": {"events": list(self._slice)},
            "pending": self.pending(),
            "result": self.result,
        }
