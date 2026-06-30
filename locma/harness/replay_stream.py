# locma/harness/replay_stream.py
from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from locma.core.actions import action_to_dict
from locma.core.engine import run_game
from locma.core.state import Phase
from locma.harness.replay_codec import cardlist_version
from locma.harness.trace import trace_hash


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
        "damage_counter": p.damage_counter,
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
    """Collects a normalized replay from run_game's recording hooks.

    Each battle step stores the *decision-point* state — the board as the acting
    seat saw it just before applying the action (``state.current == seat`` for
    every step, uniformly). This keeps the stream in one consistent perspective:
    a turn-ending Pass no longer carries the opponent's already-flipped,
    already-drawn start-of-turn snapshot, and ``write_replay``'s consecutive
    ``(seat, turn)`` grouping keeps each whole turn (closing pass included) as a
    single run with monotonic, one-per-ply turn numbers.
    """

    def __init__(self) -> None:
        self.trace: list = []
        self.draft_pool: list[list[int]] | None = None
        self.draft_picks: list[dict] = []
        self.opening: dict | None = None
        self.steps: list[dict] = []
        self.closing: dict | None = None
        self._pre: tuple[int, dict] | None = None
        self._pending: list[dict] = []

    def on_pre_step(self, seat: int, action, gs) -> None:
        # Capture the acting seat's (turn, state) BEFORE apply_battle runs. For a
        # Pass, apply_battle calls end_turn() — flipping gs.current and drawing for
        # the opponent — so only the pre-apply state is in the actor's perspective.
        self._pre = (gs.turn, snapshot(gs))

    def on_step(self, seat: int, action, gs) -> None:
        self.trace.append((seat, action))
        if isinstance(action, int):  # draft pick
            if self.draft_pool is None:
                self.draft_pool = [[c.id for c in trip] for trip in gs.draft_pool]
            self.draft_picks.append(
                {"round": len(self.draft_picks) // 2, "seat": seat, "pick": action}
            )
        else:  # battle action — record decision-point state + bucketed events
            turn, state = self._pre or (gs.turn, snapshot(gs))
            self.steps.append(
                {
                    "seat": seat,
                    "turn": turn,
                    "action": action_to_dict(action),
                    "state": state,
                    "events": self._pending,
                }
            )
            self._pre = None
            self._pending = []
            # The game-ending action's result has no later decision point to carry
            # it; record the final board so viewers can show the last move's effect.
            if gs.phase == Phase.ENDED:
                self.closing = snapshot(gs)

    def on_event(self, ev: dict) -> None:
        # Bucket only while a decision point is open; events emitted before the
        # first pre_step (start_battle's opening start_turn) are ignored — the
        # opening snapshot already shows the drawn hands.
        if self._pre is not None:
            self._pending.append(ev)

    def on_snapshot(self, gs) -> None:
        self.opening = snapshot(gs)


def _engine_version() -> str:
    try:
        return version("locma")
    except PackageNotFoundError:
        return "0+unknown"


def assemble_replay(
    rec,
    *,
    winner: int,
    turns: int,
    policy_a: str,
    policy_b: str,
    seed: int,
    a_seat: int,
    source: str,
    created_at: str | None = None,
) -> dict:
    """Build a locma-replay/3 dict from an already-populated StreamRecorder.

    Used by build_replay (after run_game) and by the interactive session driver
    (which drives the recorder's hooks live during a human game).
    """
    h = trace_hash(rec.trace, winner, turns)
    created_at = created_at or datetime.now(UTC).isoformat()
    header = {
        "replay_id": "r_" + h.split(":")[1][:12],
        "created_at": created_at,
        "source": source,
        "format": "locma-replay/3",
        "engine_version": _engine_version(),
        "cardlist_version": cardlist_version(),
        "policy_a": policy_a,
        "policy_b": policy_b,
        "seed": seed,
        "a_seat": a_seat,
        "winner": winner,
        "turns": turns,
        "step_count": len(rec.steps),
        "hash": h,
    }
    return {
        "header": header,
        "draft": {"pool": rec.draft_pool, "picks": rec.draft_picks},
        "battle": {"opening": rec.opening, "steps": rec.steps, "closing": rec.closing},
        "result": {"winner": winner, "turns": turns},
    }


def build_replay(p_a, p_b, seed, *, a_seat=0, source="ad-hoc", created_at=None) -> dict:
    p0, p1 = (p_a, p_b) if a_seat == 0 else (p_b, p_a)
    rec = StreamRecorder()
    result = run_game(
        p0,
        p1,
        seed,
        on_step=rec.on_step,
        on_snapshot=rec.on_snapshot,
        on_pre_step=rec.on_pre_step,
        on_event=rec.on_event,
    )
    return assemble_replay(
        rec,
        winner=result.winner,
        turns=result.turns,
        policy_a=p_a.name,
        policy_b=p_b.name,
        seed=seed,
        a_seat=a_seat,
        source=source,
        created_at=created_at,
    )


def build_replay_from_log_row(row: dict, *, source: str, make_policy) -> dict:
    p_a = make_policy(row["policy_a"])
    p_b = make_policy(row["policy_b"])
    rep = build_replay(p_a, p_b, row["seed"], a_seat=row["a_seat"], source=source)
    got = rep["header"]["hash"]
    if got != row.get("hash"):
        raise ValueError(f"hash mismatch: stored={row.get('hash')} got={got}")
    return rep
