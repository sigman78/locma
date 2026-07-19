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
from locma.envs.encode import (
    ACTION_SIZE,
    MAX_TOKENS,
    N_TACTICAL,
    OBS_SIZE,
    TOKEN_FEATS,
    action_mask,
    encode_battle,
    encode_battle_tokens,
    sem_index,
)
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

    Skips forced decisions (a single legal action carries no signal) and any
    action the semantic action space cannot represent (sem_index is None).
    """

    def __init__(self, teacher_seat: int, obs_mode: str = "flat", labeler=None) -> None:
        self.teacher_seat = teacher_seat
        self.obs_mode = obs_mode
        self.labeler = labeler  # optional: gs -> dict of per-state concept labels
        self.obs: list = []
        self.action: list = []
        self.mask: list = []
        self.labels: list = []
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
        if self.obs_mode == "token":
            self.obs.append(encode_battle_tokens(view))
        else:
            self.obs.append(encode_battle(view))
        self.action.append(idx)
        self.mask.append(action_mask(view, legal))
        if self.labeler is not None:
            self.labels.append(self.labeler(gs))


def record_practicum(
    teacher: str = "mcts:100",
    opponents=_DEFAULT_OPPONENTS,
    games: int = 200,
    out: str = "practicum.npz",
    seed: int = 0,
    obs_mode: str = "flat",
    labeler=None,
) -> dict:
    """Generate a practicum and write ``out`` (.npz) + its manifest.

    For each opponent and each game seed, plays BOTH seat orientations (teacher on
    seat 0 then seat 1) so the captured states are seat-balanced. Returns the
    manifest dict.

    ``obs_mode`` controls the recorded observation format:
    - ``"flat"`` (default): stores ``obs`` array of shape (n, OBS_SIZE), unchanged.
    - ``"token"``: stores four arrays (obs_tokens, obs_card_ids, obs_token_mask,
      obs_scalars) for the PPO2 tokenized observation; no ``obs`` key written.

    ``labeler``, if given, is called as ``labeler(gs)`` at every captured
    decision and must return a flat dict of floats; each key is written to the
    npz as ``concept_<key>`` (float32, one value per example).
    """
    if obs_mode not in {"flat", "token"}:
        raise ValueError(f"obs_mode must be 'flat' or 'token', got {obs_mode!r}")

    opponents = list(opponents)
    obs_all: list = []
    act_all: list = []
    mask_all: list = []
    winner_all: list = []
    seat_all: list = []
    opp_all: list = []
    gid_all: list = []
    labels_all: list = []
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
                col = _Collector(teacher_seat, obs_mode, labeler=labeler)
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
                    labels_all.extend(col.labels)
                dropped += col.dropped
                gid += 1

    n = len(act_all)
    common_arrays = dict(
        action=np.asarray(act_all, dtype=np.int64),
        mask=np.asarray(mask_all, dtype=bool).reshape(n, ACTION_SIZE),
        winner=np.asarray(winner_all, dtype=np.int8),
        seat=np.asarray(seat_all, dtype=np.int8),
        opponent_id=np.asarray(opp_all, dtype=np.int8),
        game_id=np.asarray(gid_all, dtype=np.int32),
    )
    if labeler is not None and labels_all:
        for key in labels_all[0]:
            common_arrays[f"concept_{key}"] = np.asarray(
                [d[key] for d in labels_all], dtype=np.float32
            )
    if obs_mode == "token":
        # .reshape(...) gives correct shape even when n==0 (mirrors flat branch).
        obs_arrays = dict(
            obs_tokens=(
                np.asarray([d["tokens"] for d in obs_all], dtype=np.float32).reshape(
                    n, MAX_TOKENS, TOKEN_FEATS
                )
            ),
            obs_card_ids=(
                np.asarray([d["card_ids"] for d in obs_all], dtype=np.float32).reshape(
                    n, MAX_TOKENS
                )
            ),
            obs_token_mask=(
                np.asarray([d["token_mask"] for d in obs_all], dtype=np.float32).reshape(
                    n, MAX_TOKENS
                )
            ),
            obs_scalars=(
                np.asarray([d["scalars"] for d in obs_all], dtype=np.float32).reshape(n, N_TACTICAL)
            ),
        )
    else:
        obs = np.asarray(obs_all, dtype=np.float32).reshape(n, OBS_SIZE)
        obs_arrays = dict(obs=obs)

    np.savez(out, **obs_arrays, **common_arrays)

    manifest: dict = {
        "obs_mode": obs_mode,
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
    if obs_mode == "token":
        manifest["max_tokens"] = MAX_TOKENS
        manifest["token_feats"] = TOKEN_FEATS
        manifest["n_tactical"] = N_TACTICAL
    else:
        manifest["obs_size"] = OBS_SIZE
    if labeler is not None and labels_all:
        manifest["concepts"] = sorted(labels_all[0])

    with open(_manifest_path(out), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
