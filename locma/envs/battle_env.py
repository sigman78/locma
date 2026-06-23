"""Gymnasium BattleEnv: wraps the LOCM 1.2 battle phase for RL training.

The environment handles:
  - Episode initialisation: draft (opponent drafts both seats in v1) + battle setup.
  - Observation: fixed-length float32 vector from encode_battle().
  - Action: Discrete(ACTION_SIZE) index into the canonical legal-action list.
  - Action mask: boolean array flagging valid indices (for masked-PPO etc.).
  - Reward: +1 agent win, -1 loss, 0 otherwise.
  - Termination: when gs.phase == Phase.ENDED.

The opponent policy is fixed at construction time.  In v1 it also drives the
draft for both seats.  Opponent battle turns are resolved automatically inside
reset() and step() so that the returned observation always belongs to the agent.
"""
from __future__ import annotations

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from locma.core.state import GameState, Phase
from locma.core import draft as draftmod
from locma.core import battle as battlemod
from locma.core.engine import make_battle_view, make_draft_view
from locma.data.cards_db import load_cards
from locma.envs.encode import (
    ACTION_SIZE,
    OBS_SIZE,
    action_mask,
    encode_battle,
    index_to_action,
)


class BattleEnv(gym.Env):
    """Single-agent LOCM 1.2 battle environment with a fixed opponent policy.

    Parameters
    ----------
    opponent:
        A policy object with ``draft_action(view, legal)`` and
        ``battle_action(view, legal)`` methods.
    seed:
        Base seed for reproducible episode sequences.
    agent_seat:
        Which player index (0 or 1) the RL agent controls.
    """

    metadata: dict = {}

    def __init__(self, opponent, seed: int = 0, agent_seat: int = 0) -> None:
        super().__init__()
        self.opponent = opponent
        self.base_seed = seed
        self.agent_seat = agent_seat

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SIZE)

        self._cards = load_cards()
        self._ep: int = 0  # episode counter for seed diversification
        self.gs: GameState | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _opp_play_until_agent(self) -> None:
        """Advance opponent turns until it is the agent's turn (or game ends)."""
        while (
            self.gs.phase == Phase.BATTLE
            and self.gs.current != self.agent_seat
        ):
            legal = battlemod.battle_legal(self.gs)
            view = make_battle_view(self.gs)
            action = self.opponent.battle_action(view, legal)
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

        if seed is not None:
            eff = seed
        else:
            eff = self.base_seed + self._ep
        self._ep += 1

        self.gs = GameState.new(random.Random(eff))
        draftmod.start_draft(self.gs, self._cards)

        # Opponent drafts for both seats in v1 (battle-only training target)
        while self.gs.phase == Phase.DRAFT:
            dv = make_draft_view(self.gs)
            pick = self.opponent.draft_action(dv, [0, 1, 2])
            draftmod.apply_draft_pick(self.gs, pick)

        battlemod.start_battle(self.gs)
        self._opp_play_until_agent()

        obs = encode_battle(make_battle_view(self.gs))
        return obs, {}

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask of valid action indices for the current state."""
        return action_mask(battlemod.battle_legal(self.gs))

    def step(self, idx):
        """Apply agent action and advance until the next agent decision point.

        Parameters
        ----------
        idx : int
            Index into the canonical legal-action list.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        legal = battlemod.battle_legal(self.gs)
        battlemod.apply_battle(self.gs, index_to_action(int(idx), legal))

        if self.gs.phase != Phase.ENDED:
            self._opp_play_until_agent()

        terminated = self.gs.phase == Phase.ENDED
        reward = 0.0
        if terminated:
            reward = 1.0 if self.gs.winner == self.agent_seat else -1.0

        if terminated:
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
        else:
            obs = encode_battle(make_battle_view(self.gs))

        return obs, reward, terminated, False, {}
