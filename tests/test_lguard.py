"""Tests for the exhaustive own-turn lethal solver + wrapper (E26 micro-guard 1).

Pure Python — ``find_lethal`` and ``LethalGuardBattlePolicy`` never import the
[ml] stack, so this whole file runs without the ``[ml]`` extra.
"""

from __future__ import annotations

import random

from locma.core import battle as battlemod
from locma.core.actions import Attack, Pass
from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.engine import make_battle_view
from locma.core.instance import CardInstance
from locma.core.state import GameState, Phase
from locma.policies.lguard import LethalGuardBattlePolicy, find_lethal


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


def _one_shot_lethal_state():
    """Op at 3 HP, I have one ready 5/1 attacker — a single face Attack wins."""
    gs = _gs()
    gs.players[1].health = 3
    gs.players[0].board.append(_creature(1, 5, 1))
    return gs


def _guard_then_face_lethal_state():
    """Op at 3 HP behind a 1/1 Guard; two ready 3/1 attackers. Only a 2-step
    line wins: clear the Guard, then hit face for exactly lethal."""
    gs = _gs()
    gs.players[1].health = 3
    gs.players[1].board.append(_creature(9, 1, 1, "G"))
    gs.players[0].board.append(_creature(1, 3, 1))
    gs.players[0].board.append(_creature(2, 3, 1))
    return gs


def _two_attackers_sum_to_lethal_state():
    """Op at 6 HP, no Guard; two ready 3-attack creatures — neither alone is
    lethal, but both together are. Requires a 2-action line."""
    gs = _gs()
    gs.players[1].health = 6
    gs.players[0].board.append(_creature(1, 3, 1))
    gs.players[0].board.append(_creature(2, 3, 1))
    return gs


def _no_lethal_state():
    """Op at full health with a single small attacker — no forced win."""
    gs = _gs()
    gs.players[1].health = 30
    gs.players[0].board.append(_creature(1, 2, 2))
    return gs


# ---------------------------------------------------------------------------
# find_lethal
# ---------------------------------------------------------------------------


def test_finds_one_shot_lethal():
    gs = _one_shot_lethal_state()
    line, exhausted = find_lethal(gs)
    assert exhausted is True
    assert line is not None
    assert len(line) == 1
    assert isinstance(line[0], Attack) and line[0].target_id == -1

    sim = gs
    clone_before = make_battle_view(gs)
    for a in line:
        assert a in battlemod.battle_legal(sim)
        battlemod.apply_battle(sim, a)
    assert sim.phase == Phase.ENDED and sim.winner == 0
    # the *original* state's copy used for replay was gs itself here — separately
    # verify find_lethal never mutated it before we started replaying.
    assert clone_before == make_battle_view(_one_shot_lethal_state())


def test_finds_lethal_through_guard_two_actions():
    gs = _guard_then_face_lethal_state()
    line, exhausted = find_lethal(gs)
    assert exhausted is True
    assert line is not None
    assert len(line) == 2

    for a in line:
        assert a in battlemod.battle_legal(gs)
        battlemod.apply_battle(gs, a)
    assert gs.phase == Phase.ENDED and gs.winner == 0


def test_finds_lethal_summing_two_attackers():
    gs = _two_attackers_sum_to_lethal_state()
    line, exhausted = find_lethal(gs)
    assert exhausted is True
    assert line is not None
    assert len(line) == 2

    for a in line:
        assert a in battlemod.battle_legal(gs)
        battlemod.apply_battle(gs, a)
    assert gs.phase == Phase.ENDED and gs.winner == 0


def test_no_lethal_returns_none_exhausted():
    gs = _no_lethal_state()
    line, exhausted = find_lethal(gs)
    assert line is None
    assert exhausted is True


def test_tiny_node_cap_reports_cap_hit_not_absence():
    gs = _guard_then_face_lethal_state()
    line, exhausted = find_lethal(gs, node_cap=0)
    assert line is None
    assert exhausted is False


def test_never_mutates_input_state():
    gs = _guard_then_face_lethal_state()
    before = make_battle_view(gs)
    before_health = (gs.players[0].health, gs.players[1].health)
    find_lethal(gs)
    assert make_battle_view(gs) == before
    assert (gs.players[0].health, gs.players[1].health) == before_health


def test_stats_dict_records_nodes_and_is_optional():
    gs = _guard_then_face_lethal_state()
    stats = {}
    line, exhausted = find_lethal(gs, stats=stats)
    assert line is not None
    assert stats["nodes"] > 0
    # optional: omitting stats must not raise
    find_lethal(gs)


# ---------------------------------------------------------------------------
# LethalGuardBattlePolicy
# ---------------------------------------------------------------------------


class _StubInner:
    """Records every call and always returns Pass() unless told otherwise."""

    def __init__(self):
        self.calls = 0
        self.reset_calls = 0

    def battle_action(self, view, legal, state=None):
        self.calls += 1
        return Pass()

    def reset(self, seed=None):
        self.reset_calls += 1


def test_delegates_when_no_lethal():
    gs = _no_lethal_state()
    inner = _StubInner()
    pol = LethalGuardBattlePolicy(inner)
    view = make_battle_view(gs)
    legal = battlemod.battle_legal(gs)
    action = pol.battle_action(view, legal, gs)
    assert action == Pass()
    assert inner.calls == 1
    assert pol.stats["activations"] == 0
    assert pol.stats["searches"] == 1


def test_plays_full_lethal_line_not_just_first_action():
    gs = _guard_then_face_lethal_state()
    inner = _StubInner()
    pol = LethalGuardBattlePolicy(inner)

    view = make_battle_view(gs)
    legal = battlemod.battle_legal(gs)
    a1 = pol.battle_action(view, legal, gs)
    assert isinstance(a1, Attack)
    assert inner.calls == 0  # guard handled it, never asked inner
    battlemod.apply_battle(gs, a1)

    view2 = make_battle_view(gs)
    legal2 = battlemod.battle_legal(gs)
    a2 = pol.battle_action(view2, legal2, gs)
    assert isinstance(a2, Attack) and a2.target_id == -1
    battlemod.apply_battle(gs, a2)

    assert gs.phase == Phase.ENDED and gs.winner == 0
    assert pol.stats["activations"] == 1
    assert pol.stats["searches"] == 1  # second decision popped the cached tail


def test_negative_cache_skips_research_same_turn():
    gs = _no_lethal_state()
    # add a second creature so there's more than one decision this turn
    gs.players[0].board.append(_creature(2, 1, 1))
    inner = _StubInner()
    pol = LethalGuardBattlePolicy(inner)

    view = make_battle_view(gs)
    legal = battlemod.battle_legal(gs)
    pol.battle_action(view, legal, gs)
    assert pol.stats["searches"] == 1

    # Second decision, same turn: must NOT re-search.
    view2 = make_battle_view(gs)
    legal2 = battlemod.battle_legal(gs)
    pol.battle_action(view2, legal2, gs)
    assert pol.stats["searches"] == 1
    assert inner.calls == 2


def test_reset_clears_plan_and_cache_but_not_stats():
    gs = _guard_then_face_lethal_state()
    inner = _StubInner()
    pol = LethalGuardBattlePolicy(inner)
    view = make_battle_view(gs)
    legal = battlemod.battle_legal(gs)
    pol.battle_action(view, legal, gs)
    assert pol._plan
    assert pol.stats["searches"] == 1

    pol.reset(0)
    assert pol._plan == []
    assert pol._no_lethal_turn is None
    assert inner.reset_calls == 1
    assert pol.stats["searches"] == 1  # accumulates across games, not reset


def test_probe_counts_guard_changed_move():
    gs = _one_shot_lethal_state()
    inner = _StubInner()  # always returns Pass, which differs from the lethal Attack
    pol = LethalGuardBattlePolicy(inner, probe=True)
    view = make_battle_view(gs)
    legal = battlemod.battle_legal(gs)
    action = pol.battle_action(view, legal, gs)
    assert isinstance(action, Attack)
    assert inner.calls == 1  # probe asked the inner policy too
    assert pol.stats["guard_changed_move"] == 1


def test_trivial_delegate_when_single_legal_action():
    """len(legal) == 1 (only Pass) short-circuits straight to inner, no search."""
    gs = _gs()  # empty board, no hand playables -> only Pass is legal
    inner = _StubInner()
    pol = LethalGuardBattlePolicy(inner)
    view = make_battle_view(gs)
    legal = battlemod.battle_legal(gs)
    assert len(legal) == 1
    pol.battle_action(view, legal, gs)
    assert pol.stats["searches"] == 0
    assert inner.calls == 1
