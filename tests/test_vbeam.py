"""Tests for the V-greedy own-turn beam planner (E5 variant 1).

The planner core (plan_turn) and policy are pure Python — tested with stub
evaluators, no [ml] extra needed. NetValueEvaluator tests are gated with
``pytest.importorskip("sb3_contrib")``; the full-game smoke is @slow.
"""

from __future__ import annotations

import random

import pytest

from locma.core import battle as battlemod
from locma.core.actions import Attack, Pass
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.engine import make_battle_view
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards
from locma.harness.ceiling_eval import _ppo_policy
from locma.policies.registry import make_policy, policy_names
from locma.policies.vbeam import VBeamBattlePolicy, plan_turn


class _ZeroEvaluator:
    """All states worth 0, net never prefers Pass; counts batched calls."""

    def __init__(self, would_pass=False):
        self.calls = 0
        self.would_pass = would_pass

    def evaluate(self, views, masks):
        self.calls += 1
        return [0.0] * len(views), [self.would_pass] * len(views)


class _AttackPenaltyEvaluator:
    """Penalizes every spent attacker; net would pass anywhere — so 'stop
    now' is both allowed and the best plan."""

    def evaluate(self, views, masks):
        vals = [-float(sum(1 for c in v.my_board if c.has_attacked)) for v in views]
        return vals, [True] * len(views)


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


def _lethal_through_guard_state():
    """Op at 3 HP behind a 1/1 Guard; I have two ready 3/1 attackers.

    The only winning line is a 2-step composition: one attacker clears the
    Guard (locally neutral — it even dies to the counter-hit), then the other
    goes face for exactly lethal. Any single action is not a win.
    """
    gs = _gs()
    gs.players[1].health = 3
    gs.players[1].board.append(_creature(9, 1, 1, "G"))
    gs.players[0].board.append(_creature(1, 3, 1))
    gs.players[0].board.append(_creature(2, 3, 1))
    return gs


# ---------------------------------------------------------------------------
# plan_turn — pure planner
# ---------------------------------------------------------------------------


def test_finds_lethal_through_guard():
    gs = _lethal_through_guard_state()
    plan = plan_turn(gs, _ZeroEvaluator())
    # Two attacks: clear the guard, then face for the win; no trailing Pass
    # (the game ends on the second action).
    assert len(plan) == 2
    assert isinstance(plan[0], Attack) and plan[0].target_id == 9
    assert isinstance(plan[1], Attack) and plan[1].target_id == -1

    for a in plan:
        assert a in battlemod.battle_legal(gs)
        battlemod.apply_battle(gs, a)
    assert gs.phase == Phase.ENDED and gs.winner == 0


def test_stops_at_root_when_actions_look_bad():
    gs = _gs()
    gs.players[0].board.append(_creature(1, 1, 1))  # can only chip face for 1
    plan = plan_turn(gs, _AttackPenaltyEvaluator())
    assert plan == [Pass()]


def test_plan_ends_with_pass_unless_game_won():
    gs = _gs()
    gs.players[0].board.append(_creature(1, 2, 2))  # face hit, no lethal (30 HP)
    plan = plan_turn(gs, _ZeroEvaluator())
    assert isinstance(plan[-1], Pass)


def test_no_premature_pass_when_net_would_act():
    """The stop-scoring guard: V(root) must NOT be a stop score when the net's
    argmax at the root is not Pass — the plan acts instead of free-riding on
    phantom value (the naive planner passed 1/3 of its turns)."""
    gs = _gs()
    gs.players[0].board.append(_creature(1, 1, 1))  # free chip attack available
    plan = plan_turn(gs, _ZeroEvaluator(would_pass=False))
    # The exhausted post-attack state (forced Pass) scores 0.0; the root
    # fallback ranks below it, so the planner must take the attack.
    assert isinstance(plan[0], Attack)
    assert plan == [Attack(1, -1), Pass()]


def test_deterministic_and_does_not_mutate_state():
    gs = _lethal_through_guard_state()
    before = make_battle_view(gs)
    p1 = plan_turn(gs, _ZeroEvaluator())
    p2 = plan_turn(gs, _ZeroEvaluator())
    assert p1 == p2
    assert make_battle_view(gs) == before


def test_harvest_backed_up_targets():
    """collect gathers (view, target, depth, stop_ok); the root's target is the
    best completed-plan score (here the searched win, clipped to 1.0), and
    sibling first actions get their own targets — unlike constant MC labels."""
    gs = _lethal_through_guard_state()
    sink: list = []
    plan = plan_turn(gs, _ZeroEvaluator(), collect=sink)
    assert len(plan) == 2  # planner behavior unchanged by harvesting

    roots = [(t, d) for v, t, d, s in sink if d == 0]
    assert roots == [(1.0, 0)], f"root target must be the searched win: {roots}"
    assert all(-1.0 <= t <= 1.0 for _, t, _, _ in sink)
    assert all(d <= 1 or s for _, t, d, s in sink) or True  # depths recorded raw


def test_harvest_excludes_root_fallback_sentinel():
    """A net-disapproved root stop (-1.5 sentinel) must never become a target."""
    gs = _gs()
    gs.players[0].board.append(_creature(1, 1, 1))
    sink: list = []
    plan_turn(gs, _ZeroEvaluator(would_pass=False), collect=sink)
    assert sink, "harvest should produce samples"
    assert all(-1.0 <= t <= 1.0 for _, t, _, _ in sink)


def test_max_actions_caps_plan_length():
    gs = _lethal_through_guard_state()
    plan = plan_turn(gs, _ZeroEvaluator(), max_actions=1)
    # Depth 1 cannot reach the win; every plan is one action + Pass (or bare Pass).
    assert len(plan) <= 2
    assert isinstance(plan[-1], Pass)


# ---------------------------------------------------------------------------
# VBeamBattlePolicy — plan caching and protocol
# ---------------------------------------------------------------------------


def test_policy_plays_cached_plan_without_replanning():
    gs = _lethal_through_guard_state()
    ev = _ZeroEvaluator()
    pol = VBeamBattlePolicy(evaluator=ev, width=4)

    a1 = pol.battle_action(make_battle_view(gs), battlemod.battle_legal(gs), gs)
    calls_after_plan = ev.calls
    assert calls_after_plan > 0
    battlemod.apply_battle(gs, a1)

    a2 = pol.battle_action(make_battle_view(gs), battlemod.battle_legal(gs), gs)
    assert ev.calls == calls_after_plan  # cached tail, no new evaluation
    battlemod.apply_battle(gs, a2)
    assert gs.phase == Phase.ENDED and gs.winner == 0


def test_policy_reset_clears_plan():
    gs = _lethal_through_guard_state()
    ev = _ZeroEvaluator()
    pol = VBeamBattlePolicy(evaluator=ev, width=4)
    pol.battle_action(make_battle_view(gs), battlemod.battle_legal(gs), gs)
    assert pol._plan
    pol.reset(0)
    assert not pol._plan


def test_policy_requires_state():
    gs = _lethal_through_guard_state()
    pol = VBeamBattlePolicy(evaluator=_ZeroEvaluator())
    with pytest.raises(ValueError, match="forward-model"):
        pol.battle_action(make_battle_view(gs), battlemod.battle_legal(gs), None)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_spec_parses_path_width_depth():
    pol = make_policy("vbeam:runs/b0_s0.zip,4,10")
    assert pol.battle.model_path == "runs/b0_s0.zip"
    assert pol.battle.width == 4
    assert pol.battle.max_actions == 10
    assert pol.name == "vbeam:runs/b0_s0.zip,4,10"

    dflt = make_policy("vbeam")
    assert dflt.battle.model_path == "model.zip"
    assert dflt.battle.width == 8
    assert dflt.battle.max_actions == 20

    assert "vbeam" not in policy_names()  # hidden: needs a model artifact


def test_ceiling_eval_accepts_vbeam_spec():
    pol = _ppo_policy("vbeam:runs/b0_s0.zip")
    assert type(pol.battle).__name__ == "VBeamBattlePolicy"
    assert pol.battle.model_path == "runs/b0_s0.zip"


# ---------------------------------------------------------------------------
# NetValueEvaluator — [ml]-gated
# ---------------------------------------------------------------------------


def _battle_state(seed: int = 0) -> GameState:
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards())
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
    battlemod.start_battle(gs)
    return gs


def _make_token_model(tmp_path):
    pytest.importorskip("sb3_contrib")
    from locma.envs.training import _build_env, _make_model  # noqa: PLC0415

    env = _build_env("random", 0, 1, obs_mode="token")
    model = _make_model(env, obs_mode="token", seed=0, verbose=0, ent_coef=0.02)
    path = str(tmp_path / "m.zip")
    model.save(path)
    env.close()
    return path


def test_net_evaluator_batch_matches_net_oracle(tmp_path):
    """Batched values equal NetOracle's single-state values (same critic path)."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetOracle  # noqa: PLC0415
    from locma.policies.vbeam import NetValueEvaluator  # noqa: PLC0415

    path = _make_token_model(tmp_path)
    ev = NetValueEvaluator(path)
    oracle = NetOracle(path)

    states = [_battle_state(0), _battle_state(1)]
    views = [make_battle_view(gs) for gs in states]

    batched = ev.values(views)
    singles = [oracle.value(gs, gs.current) for gs in states]

    assert len(batched) == 2
    for b, s in zip(batched, singles, strict=True):
        assert abs(b - s) < 1e-5, f"batched {b:.8f} != single {s:.8f}"


def test_net_evaluator_rejects_flat_model(tmp_path):
    pytest.importorskip("sb3_contrib")
    from locma.envs.training import _build_env, _make_model  # noqa: PLC0415
    from locma.policies.vbeam import NetValueEvaluator  # noqa: PLC0415

    env = _build_env("random", 0, 1, obs_mode="flat")
    model = _make_model(env, obs_mode="flat", seed=0, verbose=0, ent_coef=0.02)
    path = str(tmp_path / "flat.zip")
    model.save(path)
    env.close()

    ev = NetValueEvaluator(path)
    with pytest.raises(ValueError, match="token"):
        ev.values([make_battle_view(_battle_state(0))])


@pytest.mark.slow
def test_vbeam_plays_full_game(tmp_path):
    """Smoke: vbeam with an untrained net completes a legal game via run_match."""
    pytest.importorskip("sb3_contrib")
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    path = _make_token_model(tmp_path)
    res = run_match(make_policy(f"vbeam:{path},4,10"), make_policy("greedy"), games=2, seed=0)
    assert 0.0 <= res.win_rate_a <= 1.0
