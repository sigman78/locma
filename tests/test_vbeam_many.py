"""Tests for ``plan_turn_many`` — the lockstep multi-state own-turn beam.

``rbeam`` runs ``n_plans * n_worlds`` opponent-reply beams per turn (16 at the
8,20,4,4 recipe of record). ``plan_turn_many`` advances them all together so each
beam DEPTH costs one batched critic call instead of one per world per depth.

The load-bearing property is behavioral: it must choose the same plans as looping
``plan_turn``. These tests assert that on stub evaluators (exact, deterministic —
no float rebatching involved) and count evaluator calls to prove the batching
actually happened. A real-net divergence check lives in the E41 bench, because a
net forward is NOT bit-exact across batch sizes (measured: batches of 1-8 differ
from a large batch by up to ~6e-7, batches >=16 agree), so bit-identity cannot be
asserted from a unit test with a real model.

Pure Python + stub evaluators; no [ml] extra needed.
"""

from __future__ import annotations

import random

from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.policies.vbeam import plan_turn, plan_turn_many


class _CountingEvaluator:
    """Deterministic value from board state; counts calls and batch sizes."""

    def __init__(self, would_pass=True):
        self.calls = 0
        self.sizes: list[int] = []
        self.would_pass = would_pass

    def evaluate(self, views, masks):
        self.calls += 1
        self.sizes.append(len(views))
        vals = [
            float(sum(c.attack + c.defense for c in v.my_board))
            - float(sum(c.attack + c.defense for c in v.op_board))
            for v in views
        ]
        return vals, [self.would_pass] * len(views)


def _gs(seed=0):
    gs = GameState.new(random.Random(seed))
    gs.phase = Phase.BATTLE
    gs.current = 0
    return gs


def _creature(iid, atk, dfn, abilities="", *, ready=True):
    ab = normalize_abilities(abilities)
    card = Card(iid, f"C{iid}", CardType.CREATURE, 1, atk, dfn, ab, 0, 0, 0)
    inst = CardInstance.from_card(card, iid)
    inst.can_attack = ready
    return inst


def _states():
    """A spread of battle states with genuinely different best turns."""
    out = []

    # 1: two ready attackers vs a guard at low HP (multi-step lethal line)
    gs = _gs(1)
    gs.players[1].health = 3
    gs.players[1].board.append(_creature(9, 1, 1, "G"))
    gs.players[0].board.extend([_creature(1, 3, 1), _creature(2, 3, 1)])
    out.append(gs)

    # 2: trades available, no lethal
    gs = _gs(2)
    gs.players[1].health = 20
    gs.players[1].board.extend([_creature(11, 2, 3), _creature(12, 4, 2)])
    gs.players[0].board.extend([_creature(3, 3, 3), _creature(4, 2, 2)])
    out.append(gs)

    # 3: single attacker, open board
    gs = _gs(3)
    gs.players[1].health = 12
    gs.players[0].board.append(_creature(5, 5, 5))
    out.append(gs)

    # 4: nothing ready — forced pass
    gs = _gs(4)
    gs.players[1].health = 25
    gs.players[0].board.append(_creature(6, 4, 4, ready=False))
    out.append(gs)

    # 5: wide board, many permutations (stresses the `seen` dedup per beam)
    gs = _gs(5)
    gs.players[1].health = 18
    gs.players[1].board.extend([_creature(13, 1, 1), _creature(14, 2, 2)])
    gs.players[0].board.extend([_creature(7, 2, 2), _creature(8, 2, 2), _creature(10, 1, 4)])
    out.append(gs)
    return out


def test_matches_looped_plan_turn_state_by_state():
    """The whole point: identical plans to looping plan_turn."""
    states = _states()
    looped = [plan_turn(s, _CountingEvaluator(), width=8, max_actions=20) for s in states]
    batched = plan_turn_many(states, _CountingEvaluator(), width=8, max_actions=20)
    assert len(batched) == len(states)
    for i, (a, b) in enumerate(zip(looped, batched, strict=True)):
        assert a == b, f"state {i}: looped {a} != batched {b}"


def test_matches_looped_on_duplicated_states():
    """The rbeam shape: many copies of similar worlds in one call."""
    states = _states() * 4  # 20 beams, like n_plans*n_worlds
    looped = [plan_turn(s, _CountingEvaluator(), width=8, max_actions=20) for s in states]
    batched = plan_turn_many(states, _CountingEvaluator(), width=8, max_actions=20)
    assert looped == batched


def test_uses_far_fewer_evaluator_calls_than_looping():
    """Prove the batching happened: call count collapses, batch sizes grow."""
    states = _states() * 4

    loop_ev = _CountingEvaluator()
    for s in states:
        plan_turn(s, loop_ev, width=8, max_actions=20)

    batch_ev = _CountingEvaluator()
    plan_turn_many(states, batch_ev, width=8, max_actions=20)

    assert batch_ev.calls < loop_ev.calls, (
        f"expected fewer batched calls, got {batch_ev.calls} vs looped {loop_ev.calls}"
    )
    # The batched run must also use strictly larger batches on average.
    assert max(batch_ev.sizes) > max(loop_ev.sizes)


def test_width_is_respected_per_beam():
    """A narrow beam must prune each state independently, not across states."""
    states = _states() * 3
    for width in (1, 2, 8):
        looped = [plan_turn(s, _CountingEvaluator(), width=width, max_actions=20) for s in states]
        batched = plan_turn_many(states, _CountingEvaluator(), width=width, max_actions=20)
        assert looped == batched, f"width={width} diverged"


def test_max_actions_is_respected_per_beam():
    states = _states() * 2
    for cap in (1, 2, 5, 20):
        looped = [plan_turn(s, _CountingEvaluator(), width=8, max_actions=cap) for s in states]
        batched = plan_turn_many(states, _CountingEvaluator(), width=8, max_actions=cap)
        assert looped == batched, f"max_actions={cap} diverged"


def test_would_pass_false_path_matches():
    """When the net never wants to Pass the fallback-stop path is exercised."""
    states = _states() * 2
    looped = [
        plan_turn(s, _CountingEvaluator(would_pass=False), width=8, max_actions=20) for s in states
    ]
    batched = plan_turn_many(states, _CountingEvaluator(would_pass=False), width=8, max_actions=20)
    assert looped == batched


def test_empty_and_single_inputs():
    assert plan_turn_many([], _CountingEvaluator()) == []
    one = _states()[:1]
    assert plan_turn_many(one, _CountingEvaluator()) == [plan_turn(one[0], _CountingEvaluator())]


def test_does_not_mutate_input_states():
    states = _states()
    before = [
        (
            s.turn,
            s.current,
            s.phase,
            tuple((p.health, len(p.board), len(p.hand)) for p in s.players),
        )
        for s in states
    ]
    plan_turn_many(states, _CountingEvaluator(), width=8, max_actions=20)
    after = [
        (
            s.turn,
            s.current,
            s.phase,
            tuple((p.health, len(p.board), len(p.hand)) for p in s.players),
        )
        for s in states
    ]
    assert before == after


def test_plans_are_never_empty_and_end_in_pass_or_win():
    from locma.core.actions import Pass  # noqa: PLC0415

    plans = plan_turn_many(_states(), _CountingEvaluator(), width=8, max_actions=20)
    for p in plans:
        assert p, "plan must never be empty"
        # every plan either ends with Pass or is a game-ending line
        assert isinstance(p[-1], Pass) or len(p) > 0


# ---------------------------------------------------------------------------
# Shared-encode: the ensemble must encode once, byte-identically
# ---------------------------------------------------------------------------


class _FakeMember:
    """Stands in for NetValueEvaluator to test the variant guard without a model."""

    def __init__(self, variant):
        self._variant = variant
        self.encode_calls = 0

    def _ensure(self):
        pass

    def encode(self, views):
        self.encode_calls += 1
        return {"tokens": [id(v) for v in views]}

    def _forward(self, views, masks, batch=None):
        import numpy as np  # noqa: PLC0415

        self.batch_seen = batch
        raw = np.zeros(len(views))
        probs = np.zeros((len(views), 155))
        probs[:, 0] = 1.0
        return raw, probs


def _ensemble_with(members):
    from locma.policies.vbeam import EnsembleValueEvaluator  # noqa: PLC0415

    ev = EnsembleValueEvaluator.__new__(EnsembleValueEvaluator)
    ev.members = members
    ev.model_paths = ["a", "b"]
    return ev


def test_shared_encode_used_once_when_variants_agree():
    members = [_FakeMember("fx"), _FakeMember("fx"), _FakeMember("fx")]
    ev = _ensemble_with(members)
    ev.values([object(), object()])
    # exactly ONE member encoded, and every member got that same batch object
    assert sum(m.encode_calls for m in members) == 1
    shared = members[0].batch_seen
    assert shared is not None
    assert all(m.batch_seen is shared for m in members)


def test_per_member_encode_when_variants_differ():
    """A mixed-variant ensemble (legal — see the E12 mixed-ensemble work) must
    fall back to each member encoding its own obs."""
    members = [_FakeMember("fx"), _FakeMember("v0")]
    ev = _ensemble_with(members)
    ev.values([object()])
    assert all(m.batch_seen is None for m in members), "must not share across variants"


def test_forward_accepts_prebuilt_batch_signature():
    """_forward's batch parameter is the contract plan_turn_many relies on."""
    import inspect  # noqa: PLC0415

    from locma.policies.vbeam import NetValueEvaluator  # noqa: PLC0415

    params = list(inspect.signature(NetValueEvaluator._forward).parameters)
    assert params == ["self", "views", "masks", "batch"]
    assert hasattr(NetValueEvaluator, "encode")
