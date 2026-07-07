"""Gymnasium DraftEnv: the LOCM draft phase as a 30-step RL episode (E18b).

ByteRL (arXiv 2303.04096) trained draft and battle end-to-end; this env is the
staged, single-box version of that idea: learn the DRAFT with the battle phase
played out by a FROZEN battle pilot on both seats. The mirror pilot isolates
the deck signal — with identical battle play, the reward difference between
seats comes from the drafted decks alone.

Episode structure:
  - reset(): start a fresh draft; the opponent draft policy picks for the
    other seat, the agent picks for ``agent_seat``. Default LOCM draft rule
    only (both seats pick independently from the same triplet).
  - step(pick): apply the agent's pick, let the opponent pick, and when the
    30th round closes, play the battle out with ``battle_pilot`` on BOTH
    seats. Reward is the mean win (+1/-1) over ``rollouts`` battle playouts
    (rollout k>0 reshuffles both decks with a k-derived RNG — cheap variance
    reduction for the 30-picks-to-one-outcome credit assignment). All other
    steps reward 0.
  - Observation: encode_draft() — round + own-deck summary + the 3 offered
    cards (DRAFT_OBS_SIZE). Action: Discrete(3) with action_masks().

The battle playout duplicates run_game's battle loop and its safety caps
(per-turn 100, global 1000, max_turns health tiebreak) without the recording
hooks; keep the two in sync if the caps ever change.
"""

from __future__ import annotations

import copy
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
    DRAFT_OBS_SIZE,
    N_DRAFT_ACTIONS,
    draft_action_mask,
    encode_draft,
)


class DraftEnv(gym.Env):
    """Single-agent draft environment with frozen battle playout for reward.

    Parameters
    ----------
    battle_pilot:
        A battle policy object (``battle_action(view, legal, state)``) that
        plays BOTH seats in the reward playout. Frozen — never trained here.
    opponent_draft:
        A draft policy object (``draft_action(view, legal)``) drafting the
        seat the agent does not control. The incumbent to beat (balanced).
    seed:
        Base seed for reproducible episode sequences (episode k uses seed+k,
        matching BattleEnv).
    agent_seat:
        Which player index (0 or 1) the agent drafts for.
    seat_random:
        Randomize the agent's seat per episode (train as both first and
        second picker; the battle seats follow the draft seats).
    rollouts:
        Battle playouts averaged into the terminal reward. Rollout 0 uses the
        draft's own deck shuffle; each further rollout reshuffles both decks.
    """

    metadata: dict = {}

    def __init__(
        self,
        battle_pilot,
        opponent_draft,
        seed: int = 0,
        agent_seat: int = 0,
        seat_random: bool = False,
        rollouts: int = 1,
        max_turns: int = 200,
    ) -> None:
        super().__init__()
        if rollouts < 1:
            raise ValueError(f"rollouts must be >= 1, got {rollouts}")
        self.battle_pilot = battle_pilot
        self.opponent_draft = opponent_draft
        self.base_seed = seed
        self.agent_seat = agent_seat
        self.seat_random = seat_random
        self._seat_rng = random.Random(seed + 777)
        self.rollouts = rollouts
        self.max_turns = max_turns

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(DRAFT_OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_DRAFT_ACTIONS)

        self._cards = load_cards()
        self._ep: int = 0
        self._eff_seed: int = seed
        self.gs: GameState | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_obs(self) -> np.ndarray:
        return encode_draft(make_draft_view(self.gs), self.gs.picks[self.agent_seat])

    def _zero_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _opp_pick_until_agent(self) -> None:
        """Advance opponent draft picks until it is the agent's pick (or draft ends)."""
        while self.gs.phase == Phase.DRAFT and self.gs.current != self.agent_seat:
            dv = make_draft_view(self.gs)
            pick = self.opponent_draft.draft_action(dv, draftmod.draft_legal(self.gs))
            draftmod.apply_draft_pick(self.gs, pick)

    def _playout(self, gs: GameState) -> int:
        """Play the battle to completion with the frozen pilot on both seats.

        Mirrors run_game's battle loop and safety caps (see module docstring).
        Returns the winner seat.
        """
        battlemod.start_battle(gs)
        safety = 0
        while gs.phase == Phase.BATTLE and gs.turn <= self.max_turns:
            per_turn = 0
            turn_owner = gs.current
            while gs.current == turn_owner and gs.phase == Phase.BATTLE:
                legal = battlemod.battle_legal(gs)
                view = make_battle_view(gs)
                action = self.battle_pilot.battle_action(view, legal, gs)
                battlemod.apply_battle(gs, action)
                per_turn += 1
                if per_turn > 100:
                    battlemod.end_turn(gs)
                    break
            safety += 1
            if safety > 1000:
                break
        if gs.winner is None:
            h0, h1 = gs.players[0].health, gs.players[1].health
            gs.winner = 0 if h0 >= h1 else 1
        return gs.winner

    def _terminal_reward(self) -> float:
        """Mean win over `rollouts` playouts of the completed draft."""
        total = 0.0
        for k in range(self.rollouts):
            gs = copy.deepcopy(self.gs) if k < self.rollouts - 1 else self.gs
            if k > 0:
                # Fresh deck orders decorrelate the playouts; the pilot and the
                # decks stay fixed, so the mean still scores the DRAFT.
                r = random.Random(self._eff_seed * 10_007 + k)
                for p in (0, 1):
                    r.shuffle(gs.players[p].deck)
            winner = self._playout(gs)
            total += 1.0 if winner == self.agent_seat else -1.0
        return total / self.rollouts

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        eff = seed if seed is not None else self.base_seed + self._ep
        self._ep += 1
        self._eff_seed = eff
        # Mirror BattleEnv/run_game: reseed per episode so stateful policies
        # are reproducible and carry no state across episodes.
        self.opponent_draft.reset(eff)
        self.battle_pilot.reset(eff)
        if self.seat_random:
            self.agent_seat = self._seat_rng.randint(0, 1)

        self.gs = GameState.new(random.Random(eff))
        draftmod.start_draft(self.gs, self._cards)
        self._opp_pick_until_agent()
        return self._encode_obs(), {}

    def action_masks(self) -> np.ndarray:
        return draft_action_mask(draftmod.draft_legal(self.gs))

    def step(self, pick):
        draftmod.apply_draft_pick(self.gs, int(pick))
        if self.gs.phase == Phase.DRAFT:
            self._opp_pick_until_agent()

        # apply_draft_pick flips phase to BATTLE when the 30th round closes.
        terminated = self.gs.phase != Phase.DRAFT
        reward = self._terminal_reward() if terminated else 0.0
        obs = self._zero_obs() if terminated else self._encode_obs()
        return obs, reward, terminated, False, {}
