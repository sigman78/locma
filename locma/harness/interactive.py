# locma/harness/interactive.py
from __future__ import annotations

import random

from locma.core import battle as battlemod
from locma.core import draft as draftmod
from locma.core.actions import action_from_dict, action_to_dict
from locma.core.engine import make_battle_view, make_draft_view
from locma.core.state import GameState, Phase
from locma.harness.replay_stream import (
    StreamRecorder,
    _card_dict,
    _creature_dict,
    assemble_replay,
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

        if not self._battle_started:
            battlemod.start_battle(gs, emit=self._emit)
            self.rec.on_snapshot(gs)
            self._battle_started = True
            # opening draws are shown by the opening view, not animated as a slice
            self._slice = []

        self._battle_loop_until_human_or_end()

    def _battle_loop_until_human_or_end(self):
        gs = self.gs
        # per-segment backstop mirroring run_game; MAX_TURNS is the real bound on total turns
        safety = 0
        while gs.phase == Phase.BATTLE and gs.turn <= MAX_TURNS:
            if gs.current == self.human_seat:
                return  # human's battle action needed
            per_turn = 0
            turn_owner = gs.current
            while gs.current == turn_owner and gs.phase == Phase.BATTLE:
                seat = gs.current
                legal = battlemod.battle_legal(gs)
                action = self.ai.battle_action(make_battle_view(gs), legal)
                self.rec.on_pre_step(seat, action, gs)
                battlemod.apply_battle(gs, action, emit=self._emit)
                self.rec.on_step(seat, action, gs)
                per_turn += 1
                if per_turn > 100:
                    battlemod.end_turn(gs, emit=self._emit)
                    break
            safety += 1
            if safety > 1000:
                break
        self._finish()

    def submit_action(self, action_dict: dict) -> dict:
        gs = self.gs
        if self.result is not None or gs.phase != Phase.BATTLE or gs.current != self.human_seat:
            raise WrongPhase("not your battle turn")
        action = action_from_dict(action_dict)
        if action not in battlemod.battle_legal(gs):
            raise IllegalMove(f"illegal action: {action_dict!r}")
        self._slice = []
        seat = gs.current
        self.rec.on_pre_step(seat, action, gs)
        battlemod.apply_battle(gs, action, emit=self._emit)
        self.rec.on_step(seat, action, gs)
        # non-Pass keeps the same turn (gs.current still human → loop returns at once);
        # Pass flips to the AI, which is then auto-resolved.
        self._battle_loop_until_human_or_end()
        return self._response()

    def _finish(self):
        gs = self.gs
        if gs.winner is None:
            h0, h1 = gs.players[0].health, gs.players[1].health
            gs.winner = 0 if h0 >= h1 else 1
        self._replay = assemble_replay(
            self.rec,
            winner=gs.winner,
            turns=gs.turn,
            policy_a="human",
            policy_b=self.ai.name,
            seed=self.seed,
            a_seat=self.human_seat,
            source="human-vs-ai",
        )
        self.result = {
            "winner_is_human": gs.winner == self.human_seat,
            "turns": gs.turn,
            "replay_id": self._replay["header"]["replay_id"],
        }

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
