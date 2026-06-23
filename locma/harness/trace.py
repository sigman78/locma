from __future__ import annotations

import hashlib
import json

from locma.core.actions import Action, action_to_dict
from locma.core.engine import GameResult, run_game


class Recorder:
    """Collects (seat, action) steps via the engine's on_step hook."""

    def __init__(self) -> None:
        self.trace: list[tuple[int, int | Action]] = []

    def record(self, seat: int, action: int | Action, gs) -> None:
        self.trace.append((seat, action))


def record_game(policy0, policy1, seed: int, cards=None) -> tuple[GameResult, list]:
    rec = Recorder()
    result = run_game(policy0, policy1, seed, cards=cards, on_step=rec.record)
    return result, rec.trace


def _encode_step(action: int | Action) -> dict:
    if isinstance(action, int):
        return {"t": "draft", "pick": action}
    return action_to_dict(action)


def serialize_trace(trace: list) -> list[list]:
    return [[seat, _encode_step(action)] for seat, action in trace]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def trace_hash(trace: list, winner: int, turns: int) -> str:
    payload = canonical_json(serialize_trace(trace) + [winner, turns])
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_game_log(path: str, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_game_log(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
