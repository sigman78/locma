"""Single-process self-play VecEnv with batched opponent inference (E36).

STATUS: throughput NEGATIVE on a many-core box, kept as a documented negative
(default `--driver subproc`). End-to-end WITH PPO updates at matched n_envs=12,
this is SLOWER than SubprocVecEnv (batched 131 vs subproc 165 fps) and degrades
with scale (n32 69). Reason: it steps all N game engines serially in one thread,
and that serial engine cost outweighs batching the opponent forward — which lean
argmax (`_lean_masked_argmax`) already made cheap and which SubprocVecEnv already
runs N-way parallel across workers. The `scripts/e36_batched_driver.py` prototype's
~3x was measured vs a single-process SEQUENTIAL control, not vs the process-
parallel production path. It would only win on a CPU-starved / GPU-rich machine.
See docs/worklog 2026-07 "E36 training throughput". Correctness is not the issue —
the batched decisions are decision-identical (tests/test_batched_selfplay.py).

The default training path embeds the opponent forward inside each env step
(SubprocVecEnv, one-sample forwards) -- profiled as ~92% of env time. This VecEnv
runs all N games in one process and resolves every pending opponent (and draft)
decision with a single batched forward, bucketed by pool net (PFSP samples a
different frozen net per game). SB3 drives it exactly like any VecEnv and batches
the AGENT forward itself, so the training loop, rollout buffer, and gradients are
unchanged -- only the opponent side is batched.

Semantics mirror ``BattleEnv`` + ``pfsp:pool.json`` with ``draft_override``: a pool
member is sampled per game (weight-prioritised), both decks drafted by ``ldraft``,
seat randomised per episode, reward +/-1 at the game end from the agent's seat.
The batched opponent resolution is the exact logic proven decision-preserving in
``scripts/e36_batched_driver.py`` (0 mismatches vs the sequential inline loop).

``VecEnv`` (from stable_baselines3, the ``[ml]`` extra) is only imported lazily, so
this module stays import-safe without ML deps: ``_BatchedOpponentCore`` holds all
logic as a plain class and ``make_batched_opponent_vecenv`` builds the concrete
``VecEnv`` subclass on first use.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from locma.core import battle as battlemod
from locma.core import draft as draftmod
from locma.core.engine import make_battle_view, make_draft_view
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.depot import resolve_path
from locma.envs.encode import (
    ACTION_SIZE,
    action_mask,
    draft_action_mask,
    encode_battle_tokens,
    encode_draft,
    index_to_action,
    token_obs_space,
)

LDRAFT = "depot:ldraft/ldraft_s0.zip"


class _BatchedOpponentCore:
    """All driver logic + the SB3 VecEnv method surface, as a plain (import-safe)
    class. The concrete VecEnv subclass is assembled lazily in
    ``make_batched_opponent_vecenv`` so importing this module needs no ML deps."""

    def __init__(
        self,
        pool_json: str,
        n_envs: int,
        seed: int = 0,
        ldraft: str = LDRAFT,
        obs_variant: str = "fx",
    ) -> None:
        from gymnasium import spaces  # noqa: PLC0415 — lazy [ml]-adjacent dep
        from sb3_contrib import MaskablePPO  # noqa: PLC0415 — lazy [ml] dep
        from stable_baselines3.common.vec_env import VecEnv  # noqa: PLC0415

        self.variant = obs_variant
        self.cards = load_cards()
        self.ldraft = MaskablePPO.load(resolve_path(ldraft))
        self.ldraft.policy.set_training_mode(False)

        entries = json.loads(Path(pool_json).read_text())
        self.weights = [float(e.get("weight", 1.0)) for e in entries]
        self.members: list = []  # ("net", MaskablePPO) | ("script", policy)
        for e in entries:
            spec = e["spec"]
            if spec.startswith("ppo:"):
                path = spec[4:].split(",")[0].split("|")[0]
                m = MaskablePPO.load(resolve_path(path))
                m.policy.set_training_mode(False)
                self.members.append(("net", m))
            else:
                from locma.policies.registry import make_policy  # noqa: PLC0415

                self.members.append(("script", make_policy(spec).battle))

        VecEnv.__init__(self, n_envs, token_obs_space(obs_variant), spaces.Discrete(ACTION_SIZE))

        self.gs: list = [None] * n_envs
        self.seat = [0] * n_envs
        self.member = [0] * n_envs
        self.picks: list = [None] * n_envs
        self._ep = [0] * n_envs
        self._base = [seed + i * 100_000 for i in range(n_envs)]
        self._seat_rng = [random.Random(seed + 777 + i) for i in range(n_envs)]
        self._actions: np.ndarray | None = None
        self._obs_cache: list = [None] * n_envs
        self.winners: list = []

    # ------------------------------------------------------------------ helpers
    def _sample_member(self, s: int) -> int:
        return random.Random(s).choices(range(len(self.members)), weights=self.weights)[0]

    def _reset_envs(self, idxs: list[int]) -> None:
        """Sample opponent + seat, draft both decks (lockstep-batched via ldraft),
        start the battle. Does NOT resolve opponents (caller does, in bulk)."""
        if not idxs:
            return
        from locma.policies.ppo import batched_masked_argmax  # noqa: PLC0415

        for i in idxs:
            s = self._base[i] + self._ep[i]
            self.seat[i] = self._seat_rng[i].randint(0, 1)
            self.member[i] = self._sample_member(s)
            gs = GameState.new(random.Random(s))
            draftmod.start_draft(gs, self.cards, shared=False)
            self.gs[i] = gs
            self.picks[i] = []
        while any(self.gs[i].phase == Phase.DRAFT for i in idxs):
            pend = [i for i in idxs if self.gs[i].phase == Phase.DRAFT]
            obs_l, mask_l = [], []
            for i in pend:
                dv = make_draft_view(self.gs[i])
                obs_l.append(encode_draft(dv, self.picks[i]))
                mask_l.append(draft_action_mask(draftmod.draft_legal(self.gs[i])))
            picks = batched_masked_argmax(self.ldraft, obs_l, np.stack(mask_l))
            for i, p in zip(pend, picks, strict=False):
                dv = make_draft_view(self.gs[i])
                self.picks[i].append(dv.offered[int(p)])
                draftmod.apply_draft_pick(self.gs[i], int(p))
        for i in idxs:
            battlemod.start_battle(self.gs[i])

    def _resolve_opponents(self) -> None:
        """Advance every env whose turn is the opponent's until all are at the
        agent's decision or ended. Net members batched (bucketed); scripted inline."""
        from locma.policies.ppo import batched_masked_argmax  # noqa: PLC0415

        while True:
            pend = [
                i
                for i in range(self.num_envs)
                if self.gs[i].phase == Phase.BATTLE and self.gs[i].current != self.seat[i]
            ]
            if not pend:
                return
            buckets: dict[int, list[int]] = {}
            for i in pend:
                kind, obj = self.members[self.member[i]]
                if kind == "net":
                    buckets.setdefault(self.member[i], []).append(i)
                else:
                    view = make_battle_view(self.gs[i])
                    legal = battlemod.battle_legal(self.gs[i])
                    battlemod.apply_battle(self.gs[i], obj.battle_action(view, legal, self.gs[i]))
            for mem_idx, group in buckets.items():
                _, net = self.members[mem_idx]
                obs_l, mask_l, meta = [], [], []
                for i in group:
                    view = make_battle_view(self.gs[i])
                    legal = battlemod.battle_legal(self.gs[i])
                    obs_l.append(encode_battle_tokens(view, self.variant))
                    mask_l.append(action_mask(view, legal))
                    meta.append((view, legal))
                idxs = batched_masked_argmax(net, obs_l, np.stack(mask_l))
                for i, (view, legal), a in zip(group, meta, idxs, strict=False):
                    battlemod.apply_battle(self.gs[i], index_to_action(view, legal, int(a)))

    def _settle_and_reset(self) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Score any ended games, reset them (re-drafting), resolve opponents until
        every env is back at an agent decision. Returns (rewards, dones, infos).
        Only the FIRST pass sets reward/done (the step's terminal); further passes
        guard the near-impossible case of a fresh game ending before the agent moves."""
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = [{} for _ in range(self.num_envs)]
        first = True
        while True:
            ended = [i for i in range(self.num_envs) if self.gs[i].phase == Phase.ENDED]
            if not ended:
                break
            for i in ended:
                won = self.gs[i].winner == self.seat[i]
                if first:
                    rewards[i] = 1.0 if won else -1.0
                    dones[i] = True
                    infos[i]["terminal_observation"] = self._obs_cache[i]
                self.winners.append(1 if won else 0)
                self._ep[i] += 1
            self._reset_envs(ended)
            self._resolve_opponents()
            first = False
        return rewards, dones, infos

    def _mask_for(self, i: int) -> np.ndarray:
        view = make_battle_view(self.gs[i])
        return action_mask(view, battlemod.battle_legal(self.gs[i]))

    def _batched_obs(self):
        for i in range(self.num_envs):
            self._obs_cache[i] = encode_battle_tokens(make_battle_view(self.gs[i]), self.variant)
        keys = self._obs_cache[0].keys()
        return {k: np.stack([self._obs_cache[i][k] for i in range(self.num_envs)]) for k in keys}

    # --------------------------------------------------------------- VecEnv API
    def reset(self):
        self._reset_envs(list(range(self.num_envs)))
        self._resolve_opponents()
        self._batched_obs()  # populate _obs_cache before any settle
        self._settle_and_reset()  # handle any insta-ended fresh games
        return self._batched_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        for i in range(self.num_envs):
            view = make_battle_view(self.gs[i])
            legal = battlemod.battle_legal(self.gs[i])
            battlemod.apply_battle(self.gs[i], index_to_action(view, legal, int(self._actions[i])))
        self._resolve_opponents()  # opponent turns for envs the agent's move didn't end
        rewards, dones, infos = self._settle_and_reset()
        return self._batched_obs(), rewards, dones, infos

    def close(self) -> None:
        pass

    def _idx(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def get_attr(self, attr_name: str, indices=None) -> list:
        idx = self._idx(indices)
        if attr_name == "action_masks":  # sb3_contrib masking-support probe
            return [True] * len(idx)
        if attr_name == "render_mode":
            return [getattr(self, "render_mode", None)] * len(idx)
        if hasattr(self, attr_name):
            return [getattr(self, attr_name)] * len(idx)
        raise AttributeError(attr_name)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        idx = self._idx(indices)
        if method_name == "action_masks":
            return [self._mask_for(i) for i in idx]
        raise NotImplementedError(f"env_method({method_name!r}) not supported")

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(self._idx(indices))


_IMPL = None


def make_batched_opponent_vecenv(
    pool_json: str,
    n_envs: int,
    seed: int = 0,
    ldraft: str = LDRAFT,
    obs_variant: str = "fx",
):
    """Build a single-process batched-opponent self-play VecEnv (see module doc).
    The concrete class is assembled on first call so the SB3 ``VecEnv`` import stays
    lazy."""
    global _IMPL
    if _IMPL is None:
        from stable_baselines3.common.vec_env import VecEnv  # noqa: PLC0415

        class BatchedOpponentVecEnv(_BatchedOpponentCore, VecEnv):
            pass

        _IMPL = BatchedOpponentVecEnv
    return _IMPL(pool_json, n_envs, seed=seed, ldraft=ldraft, obs_variant=obs_variant)
