# locma/harness/replay_stream.py
from __future__ import annotations

from locma.core.actions import action_to_dict


def _card_dict(inst) -> dict:
    return {
        "iid": inst.instance_id,
        "card_id": inst.card.id,
        "atk": inst.attack,
        "def": inst.defense,
        "abilities": inst.abilities,
    }


def _creature_dict(inst) -> dict:
    d = _card_dict(inst)
    d["can_attack"] = inst.can_attack
    d["has_attacked"] = inst.has_attacked
    return d


def _player_dict(p) -> dict:
    return {
        "health": p.health,
        "mana": p.mana,
        "max_mana": p.max_mana,
        "next_rune": p.next_rune,
        "bonus_draw": p.bonus_draw,
        "deck_count": len(p.deck),
        "hand": [_card_dict(c) for c in p.hand],
        "board": [_creature_dict(c) for c in p.board],
    }


def snapshot(gs) -> dict:
    return {
        "current": gs.current,
        "players": [_player_dict(gs.players[0]), _player_dict(gs.players[1])],
    }


class StreamRecorder:
    """Collects a normalized replay from run_game's on_step/on_snapshot hooks."""

    def __init__(self) -> None:
        self.trace: list = []
        self.draft_pool: list[list[int]] | None = None
        self.draft_picks: list[dict] = []
        self.opening: dict | None = None
        self.steps: list[dict] = []

    def on_step(self, seat: int, action, gs) -> None:
        self.trace.append((seat, action))
        if isinstance(action, int):  # draft pick
            if self.draft_pool is None:
                self.draft_pool = [[c.id for c in trip] for trip in gs.draft_pool]
            self.draft_picks.append(
                {"round": len(self.draft_picks) // 2, "seat": seat, "pick": action}
            )
        else:  # battle action
            self.steps.append(
                {
                    "seat": seat,
                    "turn": gs.turn,
                    "action": action_to_dict(action),
                    "state": snapshot(gs),
                }
            )

    def on_snapshot(self, gs) -> None:
        self.opening = snapshot(gs)
