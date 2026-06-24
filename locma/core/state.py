from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto


class Phase(Enum):
    DRAFT = auto()
    BATTLE = auto()
    ENDED = auto()


@dataclass
class PlayerState:
    health: int = 30
    mana: int = 0
    max_mana: int = 0
    bonus_mana: int = 0  # second-player "coin": +1 mana not counted toward max
    damage_counter: int = 0  # opponent damage taken this round, mod 5 (LOCM 1.5 draw)
    bonus_draw: int = 0
    deck: list = field(default_factory=list)
    hand: list = field(default_factory=list)
    board: list = field(default_factory=list)


@dataclass
class GameState:
    rng: random.Random
    phase: Phase = Phase.DRAFT
    turn: int = 0
    current: int = 0
    players: tuple | None = None
    draft_pool: list = field(default_factory=list)
    draft_round: int = 0
    picks: tuple | None = None
    winner: int | None = None
    emit: Callable[[dict], None] | None = None  # transient event sink; None = off

    @classmethod
    def new(cls, rng: random.Random) -> GameState:
        gs = cls(rng=rng)
        gs.players = (PlayerState(), PlayerState())
        gs.picks = ([], [])
        return gs

    def opponent(self, player_idx: int) -> int:
        return 1 - player_idx

    def clone(self) -> GameState:
        new_rng = random.Random()
        new_rng.setstate(self.rng.getstate())
        return GameState(
            rng=new_rng,
            phase=self.phase,
            turn=self.turn,
            current=self.current,
            players=copy.deepcopy(self.players),
            draft_pool=list(self.draft_pool),
            draft_round=self.draft_round,
            picks=copy.deepcopy(self.picks),
            winner=self.winner,
        )
