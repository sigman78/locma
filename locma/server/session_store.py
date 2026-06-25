from __future__ import annotations

import uuid

from locma.harness.interactive import InteractiveGame


class SessionStore:
    """In-memory registry of live interactive games. Lost on restart by design."""

    def __init__(self) -> None:
        self._games: dict[str, InteractiveGame] = {}

    def create(self, *, ai_policy, seed: int, cards, rng) -> InteractiveGame:
        game_id = "g_" + uuid.uuid4().hex[:12]
        human_seat = rng.randint(0, 1)
        game = InteractiveGame(game_id, ai_policy, seed, human_seat, cards).start()
        self._games[game_id] = game
        return game

    def get(self, game_id: str) -> InteractiveGame | None:
        return self._games.get(game_id)
