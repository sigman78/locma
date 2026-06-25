"""Record a *practicum*: cheating-MCTS (or any teacher) battle decisions, for
behavior-cloning a fast policy net. Generation runs teacher-vs-baseline games and
captures one example per teacher-seat battle decision via the engine's
``on_pre_step`` hook (decision-point state, before the action is applied).

Coupled to the ``encode.py`` observation/action layout (ADR-0004): the manifest
records that layout so the distiller can refuse a stale dataset.
"""

from __future__ import annotations

import importlib.metadata
import json

import numpy as np

from locma.core.battle import battle_legal
from locma.core.engine import make_battle_view, run_game
from locma.envs.encode import ACTION_SIZE, OBS_SIZE, action_mask, encode_battle, sem_index
from locma.policies.registry import make_policy

_DEFAULT_OPPONENTS = ("random", "scripted", "greedy", "max-guard", "max-attack")


def _engine_version() -> str:
    try:
        return importlib.metadata.version("locma")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def _manifest_path(out: str) -> str:
    base = out[:-4] if out.endswith(".npz") else out
    return f"{base}.manifest.json"


class _Collector:
    """on_pre_step callback: capture (obs, index, mask) for the teacher seat.

    Skips forced decisions (a single legal action carries no signal) and overflow
    decisions whose legal index falls outside the ACTION_SIZE-wide mask.
    """

    def __init__(self, teacher_seat: int) -> None:
        self.teacher_seat = teacher_seat
        self.obs: list = []
        self.action: list = []
        self.mask: list = []
        self.dropped = 0

    def __call__(self, seat: int, action, gs) -> None:
        if seat != self.teacher_seat:
            return
        legal = battle_legal(gs)
        if len(legal) <= 1:
            return
        if action not in legal:
            return  # defensive: a well-behaved policy always returns a legal action
        view = make_battle_view(gs)
        idx = sem_index(view, action)
        if idx is None or idx >= ACTION_SIZE:
            self.dropped += 1
            return
        self.obs.append(encode_battle(view))
        self.action.append(idx)
        self.mask.append(action_mask(view, legal))


def record_practicum(
    teacher: str = "mcts:100",
    opponents=_DEFAULT_OPPONENTS,
    games: int = 200,
    out: str = "practicum.npz",
    seed: int = 0,
) -> dict:
    """Generate a practicum and write ``out`` (.npz) + its manifest.

    For each opponent and each game seed, plays BOTH seat orientations (teacher on
    seat 0 then seat 1) so the captured states are seat-balanced. Returns the
    manifest dict.
    """
    opponents = list(opponents)
    obs_all: list = []
    act_all: list = []
    mask_all: list = []
    winner_all: list = []
    seat_all: list = []
    opp_all: list = []
    gid_all: list = []
    dropped = 0
    failed_games = 0
    gid = 0

    for opp_id, opp_spec in enumerate(opponents):
        for g in range(games):
            s = seed + g
            for teacher_seat in (0, 1):
                teacher_pol = make_policy(teacher)
                opp_pol = make_policy(opp_spec)
                p0, p1 = (teacher_pol, opp_pol) if teacher_seat == 0 else (opp_pol, teacher_pol)
                col = _Collector(teacher_seat)
                try:
                    result = run_game(p0, p1, s, on_pre_step=col)
                except Exception:
                    failed_games += 1
                    gid += 1
                    continue
                k = len(col.action)
                if k:
                    obs_all.extend(col.obs)
                    act_all.extend(col.action)
                    mask_all.extend(col.mask)
                    winner_all.extend([result.winner] * k)
                    seat_all.extend([teacher_seat] * k)
                    opp_all.extend([opp_id] * k)
                    gid_all.extend([gid] * k)
                dropped += col.dropped
                gid += 1

    n = len(act_all)
    obs = np.asarray(obs_all, dtype=np.float32).reshape(n, OBS_SIZE)
    np.savez(
        out,
        obs=obs,
        action=np.asarray(act_all, dtype=np.int64),
        mask=np.asarray(mask_all, dtype=bool).reshape(n, ACTION_SIZE),
        winner=np.asarray(winner_all, dtype=np.int8),
        seat=np.asarray(seat_all, dtype=np.int8),
        opponent_id=np.asarray(opp_all, dtype=np.int8),
        game_id=np.asarray(gid_all, dtype=np.int32),
    )
    manifest = {
        "obs_size": OBS_SIZE,
        "action_size": ACTION_SIZE,
        "teacher": teacher,
        "opponents": opponents,
        "games": games,
        "seed": seed,
        "n_examples": n,
        "n_dropped_overflow": dropped,
        "failed_games": failed_games,
        "engine_version": _engine_version(),
    }
    with open(_manifest_path(out), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
