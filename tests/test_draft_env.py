"""E18b learned-draft plumbing: encoder, DraftEnv contract, registry specs."""

from dataclasses import dataclass

import numpy as np
import pytest

from locma.envs.draft_env import DraftEnv
from locma.envs.encode import (
    CARD_FEATS,
    DRAFT_OBS_SIZE,
    N_DRAFT_ACTIONS,
    N_DRAFT_SCALARS,
    draft_action_mask,
    encode_draft,
)
from locma.policies.battles import GreedyBattlePolicy
from locma.policies.drafts import BalancedDraftPolicy


@dataclass
class _CV:
    type: int
    cost: int
    attack: int
    defense: int
    abilities: str


@dataclass
class _DV:
    round: int
    offered: tuple
    taken: int | None = None


def _dv(r=0, taken=None):
    return _DV(
        r,
        (
            _CV(0, 1, 1, 1, "------"),
            _CV(2, 5, 0, -7, "------"),
            _CV(0, 3, 3, 3, "---G--"),
        ),
        taken,
    )


# ---------------------------------------------------------------------------
# encode_draft
# ---------------------------------------------------------------------------


def test_encode_draft_shape_and_scalars():
    picks = [_CV(0, 2, 2, 2, "------"), _CV(2, 7, 0, -99, "BCDGLW"), _CV(0, 2, 1, 1, "---G--")]
    obs = encode_draft(_dv(r=3), picks)
    assert obs.shape == (DRAFT_OBS_SIZE,) and obs.dtype == np.float32
    assert obs[0] == 3.0  # round
    curve = obs[1:9]
    assert curve[2] == 2.0 and curve[7] == 1.0 and curve.sum() == 3.0  # cost buckets
    types = obs[9:13]
    assert types[0] == 2.0 and types[2] == 1.0  # 2 creatures + 1 red item
    kw = obs[13:19]
    assert kw[3] == 2.0 and kw[0] == 1.0  # Guards: sentinel BCDGLW + ---G--
    assert obs[19:22].sum() == 0.0  # taken one-hot empty in the default variant


def test_encode_draft_offered_blocks_and_taken():
    obs = encode_draft(_dv(taken=1), [])
    assert obs[1:19].sum() == 0.0  # empty deck summary
    assert obs[19:22].tolist() == [0.0, 1.0, 0.0]
    # Offered card 1 (red item, cost 5, 0/-7) occupies the second card block.
    block = obs[N_DRAFT_SCALARS + CARD_FEATS : N_DRAFT_SCALARS + 2 * CARD_FEATS]
    assert block[0] == 1.0  # presence
    assert block[3] == 1.0  # red one-hot
    assert block[5] == 5.0 and block[6] == 0.0 and block[7] == -7.0


def test_draft_action_mask():
    assert draft_action_mask([0, 1, 2]).tolist() == [True, True, True]
    assert draft_action_mask([0, 2]).tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# DraftEnv
# ---------------------------------------------------------------------------


def _env(**kw):
    kw.setdefault("battle_pilot", GreedyBattlePolicy())
    kw.setdefault("opponent_draft", BalancedDraftPolicy())
    return DraftEnv(**kw)


def _run_episode(env, seed=None, pick=0):
    obs, _ = env.reset(seed=seed)
    traj = [obs]
    total = steps = 0
    terminated = False
    while not terminated:
        assert env.action_masks().all()  # default draft: all 3 always legal
        obs, r, terminated, truncated, _ = env.step(pick)
        assert not truncated
        traj.append(obs)
        total += r
        steps += 1
    return steps, total, traj


def test_draft_env_episode_contract():
    env = _env(seed=5)
    steps, total, traj = _run_episode(env)
    assert steps == 30  # exactly one agent decision per round
    assert total in (-1.0, 1.0)  # terminal-only reward
    assert traj[0].shape == (DRAFT_OBS_SIZE,)
    assert not traj[-1].any()  # terminal obs is zeros
    # Deck summary grows: after k picks the curve counts sum to k.
    assert traj[10][1:13].sum() == 2 * 10  # curve (k) + types (k)
    assert env.action_space.n == N_DRAFT_ACTIONS


def test_draft_env_deterministic_and_seed_diverse():
    a = _run_episode(_env(seed=9), seed=9)
    b = _run_episode(_env(seed=9), seed=9)
    assert a[1] == b[1]
    assert all(np.array_equal(x, y) for x, y in zip(a[2], b[2]))
    # Different seeds shuffle the draft pool differently.
    c = _run_episode(_env(seed=9), seed=10)
    assert not np.array_equal(a[2][0], c[2][0])


def test_draft_env_agent_seat_1_and_picks_tracking():
    env = _env(seed=3, agent_seat=1)
    env.reset(seed=3)
    assert len(env.gs.picks[0]) == 1 and len(env.gs.picks[1]) == 0  # opp picked first
    env.step(0)
    assert len(env.gs.picks[1]) == 1  # agent pick landed on seat 1


def test_draft_env_rollouts_mean_reward():
    env = _env(seed=7, rollouts=3)
    _, total, _ = _run_episode(env, seed=7)
    assert -1.0 <= total <= 1.0
    assert abs(total * 3 - round(total * 3)) < 1e-9  # mean of three +-1 outcomes
    with pytest.raises(ValueError):
        _env(rollouts=0)


# ---------------------------------------------------------------------------
# Registry / policy wrapper
# ---------------------------------------------------------------------------


def test_registry_learned_draft_specs():
    from locma.policies.ppo import MaskablePPODraftPolicy
    from locma.policies.registry import make_policy

    p = make_policy("ppo:model.zip,runs/draft_s0.zip")  # lazy: nothing is loaded
    assert isinstance(p.draft, MaskablePPODraftPolicy)
    assert p.draft.model_path == "runs/draft_s0.zip"
    v = make_policy("vbeam:a.zip|b.zip,8,20,runs/draft_s0.zip")
    assert isinstance(v.draft, MaskablePPODraftPolicy)
    # The float form still selects the balanced item discount (E17).
    assert make_policy("ppo:model.zip,3").draft.item_discount == 3.0
    assert make_policy("ppo:model.zip").draft.item_discount == 12.0


def test_draft_policy_note_pick_and_reset():
    from locma.policies.ppo import MaskablePPODraftPolicy

    p = MaskablePPODraftPolicy("draft.zip")
    view = _dv()
    p.note_pick(view, 2)
    assert len(p._picks) == 1 and p._picks[0].cost == 3
    p.reset()
    assert p._picks == []


@pytest.mark.slow
def test_train_draft_end_to_end(tmp_path):
    """Tiny train_draft run -> saved model loads via the registry and drafts."""
    from locma.core.engine import run_game
    from locma.envs.training import train_draft
    from locma.policies.composer import Composer
    from locma.policies.registry import make_policy

    out = str(tmp_path / "draft_tiny.zip")
    train_draft("greedy", steps=90, out=out, seed=0, n_steps=30, batch_size=30, verbose=0)
    learned = make_policy(f"ppo:model.zip,{out}").draft
    a = Composer(GreedyBattlePolicy(), learned, name="learned")
    b = Composer(GreedyBattlePolicy(), BalancedDraftPolicy(), name="balanced")
    res = run_game(a, b, seed=123)
    assert res.winner in (0, 1)
    assert len(learned._picks) == 30  # the learned draft actually drafted a deck
