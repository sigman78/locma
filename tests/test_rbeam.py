"""Tests for the reply-aware turn beam planner (``rbeam``, Priority 2).

Pure-Python core (``plan_turn_candidates`` / ``plan_turn_reply_aware`` / the
policy) is tested with stub evaluators — no [ml] extra needed. The stub scores a
state by the health gap from the view owner's perspective and reports "would
pass" when it has no ready attacker, which is enough to exercise the beam,
determinized world scoring, and the one opponent-reply ply.
"""

from __future__ import annotations

import random

import pytest

from locma.core import battle as battlemod
from locma.core.actions import Attack, Pass
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.engine import make_battle_view
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.policies.mcts import determinize
from locma.policies.rbeam import (
    RBeamBattlePolicy,
    plan_turn_reply_aware,
    score_plan_in_world,
)
from locma.policies.registry import make_policy, policy_names
from locma.policies.vbeam import _LOSS_SCORE, _WIN_SCORE, plan_turn_candidates

_CARDS = load_cards()


class _StubEvaluator:
    """Health-gap value from the view owner's perspective; counts batched calls.

    ``would_pass`` is True exactly when no ready attacker remains, so the beam
    keeps acting while attacks are available and treats an exhausted board as a
    clean stop.
    """

    def __init__(self):
        self.eval_calls = 0
        self.value_calls = 0

    @staticmethod
    def _v(view) -> float:
        return max(-1.0, min(1.0, (view.me_health - view.op_health) / 30.0))

    def evaluate(self, views, masks):
        self.eval_calls += 1
        vals = [self._v(v) for v in views]
        would_pass = [
            not any(c.can_attack and not c.has_attacked for c in v.my_board) for v in views
        ]
        return vals, would_pass

    def values(self, views):
        self.value_calls += 1
        return [self._v(v) for v in views]


def _gs():
    gs = GameState.new(random.Random(0))
    gs.phase = Phase.BATTLE
    gs.current = 0
    return gs


def _creature(iid, atk, dfn, abilities="", *, ready=True):
    ab = normalize_abilities(abilities)
    card = Card(iid, f"C{iid}", CardType.CREATURE, 1, atk, dfn, ab, 0, 0, 0)
    inst = CardInstance.from_card(card, iid)
    inst.can_attack = ready
    return inst


def _decked_state():
    """A non-terminal battle state where both players have a deck (no deck-out).

    Keeps ``start_turn`` draws (triggered when a plan ends its turn with Pass)
    away from the 10-per-miss fatigue edge, so world scoring is clean.
    """
    gs = _gs()
    gs.players[0].health = 30
    gs.players[1].health = 20
    for pid in (0, 1):
        gs.players[pid].deck = [_creature(500 + pid * 10 + i, 1, 1) for i in range(5)]
    return gs


def _lethal_through_guard_state():
    """Op at 3 HP behind a 1/1 Guard; two ready 3/1 attackers -> a 2-step lethal."""
    gs = _gs()
    gs.players[1].health = 3
    gs.players[1].board.append(_creature(9, 1, 1, "G"))
    gs.players[0].board.append(_creature(1, 3, 1))
    gs.players[0].board.append(_creature(2, 3, 1))
    return gs


# ---------------------------------------------------------------------------
# plan_turn_candidates — top-k own-turn plans
# ---------------------------------------------------------------------------


def test_candidates_are_distinct_and_best_first():
    gs = _lethal_through_guard_state()
    cands = plan_turn_candidates(gs, _StubEvaluator(), k=4)
    assert cands, "must return at least the root stop"
    plans = [tuple(p) for _s, p in cands]
    assert len(plans) == len(set(plans)), "plans must be deduplicated"
    scores = [s for s, _p in cands]
    assert scores == sorted(scores, reverse=True), "best score first"
    assert scores[0] == _WIN_SCORE, "the lethal line is the top candidate here"


def test_candidates_respects_k():
    gs = _lethal_through_guard_state()
    assert len(plan_turn_candidates(gs, _StubEvaluator(), k=1)) == 1


# ---------------------------------------------------------------------------
# score_plan_in_world — one expectiminimax leaf
# ---------------------------------------------------------------------------


def test_score_world_returns_win_sentinel_for_lethal_line():
    gs = _lethal_through_guard_state()
    ev = _StubEvaluator()
    det = determinize(gs, random.Random(0), _CARDS)
    lethal = [Attack(1, 9), Attack(2, -1)]
    score = score_plan_in_world(det, lethal, 0, ev, ev, width=8, max_actions=20)
    assert score == _WIN_SCORE


def test_score_world_runs_opponent_reply_then_critic():
    gs = _decked_state()
    gs.players[0].board.append(_creature(1, 2, 2))
    ev = _StubEvaluator()
    det = determinize(gs, random.Random(1), _CARDS)
    score = score_plan_in_world(det, [Pass()], 0, ev, ev, width=4, max_actions=10)
    assert ev.eval_calls > 0, "the opponent's own-turn beam must have run"
    assert ev.value_calls > 0, "the back-to-us state must be scored by the critic"
    assert _LOSS_SCORE <= score <= _WIN_SCORE


def test_score_world_does_not_mutate_the_root():
    gs = _decked_state()
    gs.players[0].board.append(_creature(1, 2, 2))
    before = make_battle_view(gs)
    det = determinize(gs, random.Random(2), _CARDS)
    ev = _StubEvaluator()
    score_plan_in_world(det, [Pass()], 0, ev, ev, width=4, max_actions=10)
    assert make_battle_view(gs) == before


# ---------------------------------------------------------------------------
# plan_turn_reply_aware — root plan selection
# ---------------------------------------------------------------------------


def test_returns_lethal_immediately():
    gs = _lethal_through_guard_state()
    plan = plan_turn_reply_aware(
        gs, _StubEvaluator(), cards=_CARDS, rng=random.Random(0), n_plans=4, n_worlds=3
    )
    assert len(plan) == 2 and all(isinstance(a, Attack) for a in plan)


def test_prefers_face_damage_over_passing():
    gs = _decked_state()
    gs.players[0].board.append(_creature(1, 3, 1))  # one ready attacker, no op board
    plan = plan_turn_reply_aware(
        gs, _StubEvaluator(), cards=_CARDS, rng=random.Random(0), n_plans=4, n_worlds=2
    )
    assert isinstance(plan[0], Attack) and plan[0].target_id == -1


def test_reply_aware_is_deterministic_given_seed():
    gs = _decked_state()
    gs.players[0].board.append(_creature(1, 3, 1))
    kw = {"cards": _CARDS, "n_plans": 3, "n_worlds": 3}
    p1 = plan_turn_reply_aware(gs, _StubEvaluator(), rng=random.Random(7), **kw)
    p2 = plan_turn_reply_aware(gs, _StubEvaluator(), rng=random.Random(7), **kw)
    assert p1 == p2


# ---------------------------------------------------------------------------
# RBeamBattlePolicy — plan caching and protocol
# ---------------------------------------------------------------------------


def test_policy_plays_cached_plan_without_replanning():
    gs = _lethal_through_guard_state()
    ev = _StubEvaluator()
    pol = RBeamBattlePolicy(evaluator=ev, width=4, n_plans=2, n_worlds=1)

    a1 = pol.battle_action(make_battle_view(gs), battlemod.battle_legal(gs), gs)
    calls = ev.eval_calls + ev.value_calls
    assert calls > 0
    battlemod.apply_battle(gs, a1)

    a2 = pol.battle_action(make_battle_view(gs), battlemod.battle_legal(gs), gs)
    assert ev.eval_calls + ev.value_calls == calls, "cached tail must not re-plan"
    battlemod.apply_battle(gs, a2)
    assert gs.phase == Phase.ENDED and gs.winner == 0


def test_policy_reset_clears_plan_and_reseeds():
    gs = _lethal_through_guard_state()
    pol = RBeamBattlePolicy(evaluator=_StubEvaluator(), width=4, n_plans=2, n_worlds=1, seed=3)
    pol.battle_action(make_battle_view(gs), battlemod.battle_legal(gs), gs)
    assert pol._plan
    pol.reset()
    assert not pol._plan


def test_policy_requires_state():
    gs = _lethal_through_guard_state()
    pol = RBeamBattlePolicy(evaluator=_StubEvaluator())
    with pytest.raises(ValueError, match="forward-model"):
        pol.battle_action(make_battle_view(gs), battlemod.battle_legal(gs), None)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_spec_parses_all_params():
    pol = make_policy("rbeam:runs/b0_s0.zip,4,10,3,2")
    assert pol.battle.model_path == "runs/b0_s0.zip"
    assert pol.battle.width == 4
    assert pol.battle.max_actions == 10
    assert pol.battle.n_plans == 3
    assert pol.battle.n_worlds == 2
    assert pol.name == "rbeam:runs/b0_s0.zip,4,10,3,2"


def test_registry_defaults_and_hidden():
    dflt = make_policy("rbeam")
    assert dflt.battle.model_path == "model.zip"
    assert dflt.battle.width == 8
    assert dflt.battle.max_actions == 20
    assert dflt.battle.n_plans == 4
    assert dflt.battle.n_worlds == 4
    assert "rbeam" not in policy_names()  # hidden: needs a model artifact
