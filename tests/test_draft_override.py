"""E19 plumbing: BothSeatsDraftPolicy routing + training-env draft_override."""

from dataclasses import dataclass

import pytest

from locma.policies.drafts import (
    BalancedDraftPolicy,
    BothSeatsDraftPolicy,
    PartialRandomDraftPolicy,
)


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


def _view(r, taken=None):
    return _DV(r, (_CV(0, 1, 1, 1, "------"),) * 3, taken)


class _Spy:
    name = "spy"

    def __init__(self):
        self.calls: list[int] = []  # rounds seen
        self.noted: list[int] = []

    def draft_action(self, view, legal):
        self.calls.append(view.round)
        return 0

    def note_pick(self, view, idx):
        self.noted.append(view.round)

    def reset(self, seed=None):
        self.calls = []
        self.noted = []


def test_both_seats_routes_alternate_picks_per_round():
    a, b = _Spy(), _Spy()
    p = BothSeatsDraftPolicy(a, b)
    for r in range(30):
        p.draft_action(_view(r), [0, 1, 2])  # seat 0 (first pick of the round)
        p.draft_action(_view(r), [0, 1, 2])  # seat 1
    assert a.calls == list(range(30)) and b.calls == list(range(30))


def test_both_seats_stateful_children_track_separate_decks():
    a, b = BalancedDraftPolicy(), BalancedDraftPolicy()
    p = BothSeatsDraftPolicy(a, b)
    for r in range(5):
        p.draft_action(_view(r), [0, 1, 2])
        p.draft_action(_view(r), [0, 1, 2])
    assert len(a._picks) == 5 and len(b._picks) == 5  # not 10-card mixes


def test_both_seats_reset_and_shared_draft_guard():
    a, b = _Spy(), _Spy()
    p = BothSeatsDraftPolicy(a, b, name="x")
    p.draft_action(_view(0), [0, 1, 2])
    p.reset(3)
    assert a.calls == [] and p._round == -1
    p.draft_action(_view(0), [0, 1, 2])  # routing restarts at `first`
    assert a.calls == [0] and b.calls == []
    with pytest.raises(ValueError):
        p.draft_action(_view(1, taken=0), [1, 2])  # shared-draft view


def test_both_seats_composes_with_partial_random():
    # PartialRandom calls note_pick (not draft_action) on overridden rounds; the
    # router must advance identically so seats stay aligned afterwards.
    a, b = _Spy(), _Spy()
    p = PartialRandomDraftPolicy(BothSeatsDraftPolicy(a, b), k=4, seed=7)
    p.reset(7)
    for r in range(30):
        p.draft_action(_view(r), [0, 1, 2])
        p.draft_action(_view(r), [0, 1, 2])
    assert sorted(a.calls + a.noted) == list(range(30))
    assert sorted(b.calls + b.noted) == list(range(30))
    assert len(a.noted) == len(b.noted) == 4  # k random rounds hit both seats


def test_draft_override_policy_resolution():
    from locma.envs.training import _draft_override_policy  # noqa: PLC0415

    named = _draft_override_policy("balanced")
    assert isinstance(named.first, BalancedDraftPolicy)
    assert named.first is not named.second  # independent per-seat state
    from locma.policies.ppo import MaskablePPODraftPolicy  # noqa: PLC0415

    learned = _draft_override_policy("runs/ldraft_s0.zip")  # lazy: nothing loaded
    assert isinstance(learned.first, MaskablePPODraftPolicy)
    with pytest.raises(ValueError):
        _draft_override_policy("no-such-draft.txt")  # neither named nor a model path


def test_draft_override_policy_values_json(tmp_path):
    """E31a: a values-keyed JSON table loads as a per-seat DistilledDraftPolicy."""
    import json  # noqa: PLC0415

    from locma.envs.training import _draft_override_policy  # noqa: PLC0415
    from locma.policies.drafts import DistilledDraftPolicy  # noqa: PLC0415

    p = tmp_path / "table.json"
    p.write_text(json.dumps({"values": {"1": 5.0, "2": 1.0}, "w_need": 3.0, "w_creature": 2.0}))
    over = _draft_override_policy(str(p))
    assert isinstance(over.first, DistilledDraftPolicy)
    assert over.first is not over.second  # independent per-seat state
    assert over.first.values == {1: 5.0, 2: 1.0}


def test_battle_env_draft_override_changes_decks():
    pytest.importorskip("gymnasium")  # ML-only (BattleEnv)
    from locma.envs.training import _make_battle_env  # noqa: PLC0415

    env = _make_battle_env("greedy", seed=0, draft_override="random")
    env.reset(seed=5)
    plain = _make_battle_env("greedy", seed=0)
    plain.reset(seed=5)
    assert env.opponent.draft.name == "override:random"
    d_over = [c.card.id for c in env.gs.players[0].deck]
    d_plain = [c.card.id for c in plain.gs.players[0].deck]
    assert d_over != d_plain  # the override actually drafted the decks
    assert (
        len(env.gs.players[0].deck) + len(env.gs.players[0].hand) + len(env.gs.players[0].board)
        == 30
    )


def test_battle_env_draft_override_shared_draft_rejected():
    pytest.importorskip("gymnasium")
    from locma.envs.training import _make_battle_env  # noqa: PLC0415

    with pytest.raises(ValueError):
        _make_battle_env("greedy", seed=0, shared_draft=True, draft_override="balanced")


@pytest.mark.slow
def test_train_agent_with_learned_draft_override(tmp_path):
    """E19 path end-to-end: battle training on decks drafted by the ldraft net."""
    pytest.importorskip("sb3_contrib")
    from locma.envs.training import train_agent  # noqa: PLC0415

    out = str(tmp_path / "e19_tiny.zip")
    train_agent(
        "greedy",
        steps=60,
        out=out,
        seed=0,
        n_steps=30,
        batch_size=30,
        verbose=0,
        draft_override="depot:ldraft/ldraft_s0.zip",
    )
    import os  # noqa: PLC0415

    assert os.path.exists(out)
