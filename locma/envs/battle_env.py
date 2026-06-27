"""Gymnasium BattleEnv: wraps the LOCM 1.2 battle phase for RL training.

The environment handles:
  - Episode initialisation: draft (opponent drafts both seats in v1) + battle setup.
  - Observation: fixed-length float32 vector (OBS_SIZE) from encode_battle().
  - Action: Discrete(ACTION_SIZE) — a fixed *semantic* slot index (Pass / Summon /
    Use / Attack), mapped back to a concrete Action via index_to_action().
  - Action mask: boolean array flagging *which* semantic actions are legal now
    (built from the real legal list — for masked-PPO etc.).
  - Reward: +1 agent win, -1 loss, 0 otherwise.
  - Termination: when gs.phase == Phase.ENDED.

The opponent policy is fixed at construction time.  In v1 it also drives the
draft for both seats.  Opponent battle turns are resolved automatically inside
reset() and step() so that the returned observation always belongs to the agent.
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from locma.core import battle as battlemod
from locma.core import draft as draftmod
from locma.core.engine import make_battle_view, make_draft_view
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.envs.encode import (
    ACTION_SIZE,
    OBS_SIZE,
    action_mask,
    encode_battle,
    index_to_action,
)


def _encoder(obs_mode: str):
    if obs_mode == "base":
        return OBS_SIZE, lambda view, legal: encode_battle(view)
    if obs_mode == "tactical":
        from locma.envs.encode_tactical import OBS_SIZE as TACTICAL_OBS_SIZE  # noqa: PLC0415
        from locma.envs.encode_tactical import encode_battle as encode_tactical  # noqa: PLC0415

        return TACTICAL_OBS_SIZE, encode_tactical
    raise ValueError(f"unknown obs_mode {obs_mode!r}")


class BattleEnv(gym.Env):
    """Single-agent LOCM 1.2 battle environment with a fixed opponent policy.

    Parameters
    ----------
    opponent:
        A policy object with ``draft_action(view, legal)`` and
        ``battle_action(view, legal, state)`` methods. BattleEnv passes the
        full forward-model ``state`` (as the play harness does, see
        ``engine.py``), so search opponents (``mcts``, ``azlite``, ``dmcts``)
        work as training opponents; heuristic opponents ignore the argument.
    seed:
        Base seed for reproducible episode sequences.
    agent_seat:
        Which player index (0 or 1) the RL agent controls.
    """

    metadata: dict = {}

    def __init__(
        self,
        opponent,
        seed: int = 0,
        agent_seat: int = 0,
        seat_random: bool = False,
        obs_mode: str = "base",
        reward_mode: str = "sparse",
        reward_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.opponent = opponent
        self.base_seed = seed
        self.agent_seat = agent_seat
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale
        self._obs_size, self._encode_battle = _encoder(obs_mode)
        if reward_mode not in {"sparse", "health", "board"}:
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        # seat_random: randomize the agent's seat per episode so it trains as both
        # first AND second player (the seat-2 coin/tempo openings) — eval is mirrored
        # across both seats, so seat-0-only training is a coverage gap.
        self.seat_random = seat_random
        self._seat_rng = random.Random(seed + 777)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SIZE)

        self._cards = load_cards()
        self._ep: int = 0  # episode counter for seed diversification
        self.gs: GameState | None = None

    def _obs(self) -> np.ndarray:
        legal = battlemod.battle_legal(self.gs)
        return self._encode_battle(make_battle_view(self.gs), legal)

    def _potential(self) -> float:
        """Small dense reward potential from the agent seat's perspective."""
        me = self.gs.players[self.agent_seat]
        op = self.gs.players[1 - self.agent_seat]
        h = (me.health - op.health) / 30.0
        if self.reward_mode == "health":
            return h
        b = (
            sum(c.attack + c.defense for c in me.board)
            - sum(c.attack + c.defense for c in op.board)
        ) / 20.0
        return h + 0.5 * b

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _opp_play_until_agent(self) -> None:
        """Advance opponent turns until it is the agent's turn (or game ends)."""
        while self.gs.phase == Phase.BATTLE and self.gs.current != self.agent_seat:
            legal = battlemod.battle_legal(self.gs)
            view = make_battle_view(self.gs)
            action = self.opponent.battle_action(view, legal, self.gs)
            battlemod.apply_battle(self.gs, action)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Start a new episode.

        Returns
        -------
        obs : np.ndarray of shape (OBS_SIZE,)
        info : dict
        """
        super().reset(seed=seed)

        eff = seed if seed is not None else self.base_seed + self._ep
        self._ep += 1
        if self.seat_random:
            self.agent_seat = self._seat_rng.randint(0, 1)

        self.gs = GameState.new(random.Random(eff))
        draftmod.start_draft(self.gs, self._cards)

        # Opponent drafts for both seats in v1 (battle-only training target)
        while self.gs.phase == Phase.DRAFT:
            dv = make_draft_view(self.gs)
            pick = self.opponent.draft_action(dv, [0, 1, 2])
            draftmod.apply_draft_pick(self.gs, pick)

        battlemod.start_battle(self.gs)
        self._opp_play_until_agent()

        obs = self._obs()
        return obs, {}

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask of which semantic actions are legal right now."""
        return action_mask(make_battle_view(self.gs), battlemod.battle_legal(self.gs))

    def step(self, idx):
        """Apply agent action and advance until the next agent decision point.

        Parameters
        ----------
        idx : int
            A semantic action-space index (see encode.py); index_to_action maps it
            to the concrete legal Action.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        legal = battlemod.battle_legal(self.gs)
        view = make_battle_view(self.gs)
        before = self._potential() if self.reward_mode != "sparse" else 0.0
        battlemod.apply_battle(self.gs, index_to_action(view, legal, int(idx)))

        if self.gs.phase != Phase.ENDED:
            self._opp_play_until_agent()

        terminated = self.gs.phase == Phase.ENDED
        reward = 0.0
        if terminated:
            reward = 1.0 if self.gs.winner == self.agent_seat else -1.0
        elif self.reward_mode != "sparse":
            reward = self.reward_scale * (self._potential() - before)

        if terminated:
            obs = np.zeros(self._obs_size, dtype=np.float32)
        else:
            obs = self._obs()

        return obs, reward, terminated, False, {}
