from __future__ import annotations

import pytest

gym = pytest.importorskip("gymnasium")

import numpy as np  # noqa: E402

from locma.envs.battle_env import BattleEnv  # noqa: E402
from locma.envs.encode import OBS_SIZE, encode_battle  # noqa: E402
from locma.policies.battles import RandomBattlePolicy  # noqa: E402
from locma.policies.composer import Composer  # noqa: E402
from locma.policies.drafts import RandomDraftPolicy  # noqa: E402


def test_env_reset_step():
    opp = Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name="opp")
    env = BattleEnv(opponent=opp, seed=0)
    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    mask = env.action_masks()
    assert mask.any()
    idx = int(np.argmax(mask))
    obs, reward, terminated, truncated, info = env.step(idx)
    assert reward in (-1.0, 0.0, 1.0)


def test_battle_env_seat_random_covers_both_seats():
    opp = Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name="opp")
    env = BattleEnv(opponent=opp, seed=0, seat_random=True)
    seats = set()
    for _ in range(30):
        env.reset()
        seats.add(env.agent_seat)
    assert seats == {0, 1}  # both first- and second-player seats exercised


def test_battle_env_seat_fixed_by_default():
    opp = Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name="opp")
    env = BattleEnv(opponent=opp, seed=0)  # seat_random defaults False
    for _ in range(5):
        env.reset()
        assert env.agent_seat == 0


def test_search_opponent_works_as_training_opponent():
    """A search opponent (azlite) requires the forward-model ``state``; the env
    must pass it (mirrors the play harness). Without it azlite raises ValueError."""
    from locma.policies.registry import make_policy  # noqa: PLC0415

    opp = make_policy("azlite:16")  # low iters: fast, still exercises the search path
    env = BattleEnv(opponent=opp, seed=0)
    obs, info = env.reset()  # opponent may take turns here — needs state, must not raise
    assert obs.shape[0] == env.observation_space.shape[0]
    mask = env.action_masks()
    obs, reward, terminated, truncated, info = env.step(int(np.argmax(mask)))
    assert reward in (-1.0, 0.0, 1.0)


class _RecordingOpponent:
    """Wraps a working policy but records every `reset(seed)` call, so tests
    can assert BattleEnv.reset() reseeds the opponent (mirrors run_game,
    engine.py, which resets both policies per game for determinism)."""

    def __init__(self):
        self._inner = Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name="opp")
        self.reset_calls: list = []

    def draft_action(self, view, legal):
        return self._inner.draft_action(view, legal)

    def battle_action(self, view, legal, state=None):
        return self._inner.battle_action(view, legal, state)

    def reset(self, seed=None):
        self.reset_calls.append(seed)
        self._inner.reset(seed)


def test_battle_env_reset_resets_opponent_policy():
    opp = _RecordingOpponent()
    env = BattleEnv(opponent=opp, seed=0)

    env.reset()
    assert opp.reset_calls == [0]  # base_seed(0) + ep(0)

    env.reset()
    assert opp.reset_calls == [0, 1]  # base_seed(0) + ep(1)

    env.reset(seed=42)
    assert opp.reset_calls == [0, 1, 42]  # explicit seed passed through


def test_obs_size_matches_encode():
    """Verify OBS_SIZE constant matches encode_battle output length."""
    from locma.core.views import BattleView, CardView  # noqa: PLC0415

    dummy_card = CardView(
        instance_id=1, card_id=1, type=0, cost=1, attack=2, defense=3, abilities="------"
    )
    view = BattleView(
        turn=1,
        me_health=30,
        me_mana=1,
        op_health=30,
        op_hand_count=5,
        my_hand=(dummy_card,),
        my_board=(dummy_card,),
        op_board=(dummy_card,),
    )
    encoded = encode_battle(view)
    assert len(encoded) == OBS_SIZE, f"encode_battle returned {len(encoded)}, expected {OBS_SIZE}"
    assert encoded.dtype == np.float32
