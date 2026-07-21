"""E33 board-potential reward shaping (BattleEnv) — pure-engine tests, no [ml].

Potential-based shaping r += w*(gamma*Phi(s') - Phi(s)) with Phi(terminal)=0 is
optimal-policy-preserving: over one episode the shaping telescopes to
gamma^T*Phi(s_T) - Phi(s_0). With agent_seat=0 the agent moves first from empty
boards, so Phi(s_0)=0, and at gamma=1 the total shaping is exactly 0 — i.e. the
episode's summed reward is unchanged (still the terminal +/-1). That invariant
plus "the per-step rewards actually differ from the w=0 run" pins the wiring.
"""

from __future__ import annotations

import numpy as np

from locma.envs.battle_env import BattleEnv
from locma.policies.registry import make_policy


def _rollout(weight: float, gamma: float = 1.0, seed: int = 7):
    """Play one episode taking the first legal action each step; return the reward
    sequence. Deterministic: same seed + same action rule => identical transitions,
    so runs at different weights are directly comparable step-for-step."""
    env = BattleEnv(
        opponent=make_policy("scripted"),
        seed=seed,
        agent_seat=0,
        seat_random=False,
        obs_mode="token-fx",
        board_potential_weight=weight,
        shaping_gamma=gamma,
    )
    env.reset(seed=seed)
    rewards = []
    done = False
    while not done:
        idx = int(np.argmax(env.action_masks()))  # first legal semantic action
        _, r, term, trunc, _ = env.step(idx)
        rewards.append(r)
        done = term or trunc
    return rewards


def test_shaping_off_is_sparse_terminal():
    r = _rollout(0.0)
    assert all(x == 0.0 for x in r[:-1]), "unshaped reward must be 0 until termination"
    assert abs(r[-1]) == 1.0, "terminal reward must be +/-1"


def test_shaping_telescopes_to_zero_at_gamma_one():
    base = _rollout(0.0)
    shaped = _rollout(0.5, gamma=1.0)
    assert len(base) == len(shaped), "same seed+actions must give the same trajectory length"
    # potential-based, Phi(s_0)=Phi(terminal)=0 at gamma=1 -> summed reward unchanged
    assert abs(sum(shaped) - sum(base)) < 1e-6
    # ...but the per-step credit is genuinely reshaped (shaping is actually active)
    assert any(abs(a - b) > 1e-9 for a, b in zip(base, shaped, strict=True))
