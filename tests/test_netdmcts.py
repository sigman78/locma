"""Tests for NetGuidedDMCTSBattlePolicy.

Requires the [ml] extra (sb3_contrib, torch). All tests are gated with
``pytest.importorskip("sb3_contrib")`` so they are skipped cleanly in CI
without the extra.
"""

from __future__ import annotations

import random

import pytest

from locma.core import battle as battlemod
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _battle_state(seed: int = 0) -> GameState:
    """Return a GameState at a battle decision point with >1 legal action."""
    gs = GameState.new(random.Random(seed))
    start_draft(gs, load_cards())
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
    battlemod.start_battle(gs)
    return gs


def _make_token_model(tmp_path):
    """Build and save an untrained token MaskablePPO; return the path."""
    pytest.importorskip("sb3_contrib")
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")

    from locma.envs.training import _build_env, _make_model  # noqa: PLC0415

    env = _build_env("random", 0, 1, obs_mode="token")
    model = _make_model(env, obs_mode="token", seed=0, verbose=0, ent_coef=0.02)
    path = str(tmp_path / "m.zip")
    model.save(path)
    env.close()
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_netdmcts_returns_legal_action(tmp_path):
    """battle_action returns an action from the legal set (small K, I for speed)."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetGuidedDMCTSBattlePolicy  # noqa: PLC0415

    model_path = _make_token_model(tmp_path)
    pol = NetGuidedDMCTSBattlePolicy(
        model_path=model_path, determinizations=2, iterations=8, seed=0
    )

    gs = _battle_state(seed=0)
    legal = list(battlemod.battle_legal(gs))
    assert len(legal) > 1, "test needs >1 legal action"

    from locma.core.engine import make_battle_view  # noqa: PLC0415

    view = make_battle_view(gs)
    action = pol.battle_action(view, legal, state=gs)
    assert action in legal, f"returned action {action!r} not in legal {legal}"


def test_netdmcts_raises_without_state(tmp_path):
    """battle_action raises ValueError when state=None."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetGuidedDMCTSBattlePolicy  # noqa: PLC0415

    model_path = _make_token_model(tmp_path)
    pol = NetGuidedDMCTSBattlePolicy(model_path=model_path, determinizations=2, iterations=8)

    gs = _battle_state(seed=0)
    legal = list(battlemod.battle_legal(gs))
    from locma.core.engine import make_battle_view  # noqa: PLC0415

    view = make_battle_view(gs)

    with pytest.raises(ValueError, match="state"):
        pol.battle_action(view, legal, state=None)


def test_netdmcts_single_legal_action_no_search(tmp_path):
    """When only one legal action exists, it is returned immediately (no search)."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetGuidedDMCTSBattlePolicy  # noqa: PLC0415

    # Use a non-existent model path to prove no model is loaded
    pol = NetGuidedDMCTSBattlePolicy(
        model_path="nonexistent_model.zip", determinizations=2, iterations=8
    )

    gs = _battle_state(seed=0)
    from locma.core.engine import make_battle_view  # noqa: PLC0415

    view = make_battle_view(gs)
    # Provide a single-element legal list
    from locma.core.actions import Pass  # noqa: PLC0415

    single = [Pass()]
    result = pol.battle_action(view, single, state=gs)
    assert result == single[0]
    # Model should not have been loaded
    assert pol._oracle._model is None


def test_netdmcts_deterministic_stable(tmp_path):
    """deterministic=True: two calls on the same (view, legal, state) return the same action."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetGuidedDMCTSBattlePolicy  # noqa: PLC0415

    model_path = _make_token_model(tmp_path)
    pol = NetGuidedDMCTSBattlePolicy(
        model_path=model_path,
        determinizations=2,
        iterations=8,
        seed=0,
        deterministic=True,
    )

    gs = _battle_state(seed=0)
    legal = list(battlemod.battle_legal(gs))
    assert len(legal) > 1, "test needs >1 legal action"

    from locma.core.engine import make_battle_view  # noqa: PLC0415

    view = make_battle_view(gs)

    a1 = pol.battle_action(view, legal, state=gs)
    a2 = pol.battle_action(view, legal, state=gs)
    assert a1 == a2, f"deterministic=True gave different actions: {a1!r} vs {a2!r}"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_netdmcts_in_registry():
    """make_policy('netdmcts:...') parses params and pairs balanced draft."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
    from locma.policies.net_oracle import NetGuidedDMCTSBattlePolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    p = make_policy("netdmcts:15,80,1.5,runs/x.zip")
    assert isinstance(p.battle, NetGuidedDMCTSBattlePolicy)
    assert isinstance(p.draft, BalancedDraftPolicy)
    assert p.battle.K == 15
    assert p.battle.iterations == 80
    assert p.battle.c_puct == 1.5
    assert p.battle.model_path == "runs/x.zip"


def test_netdmcts_registry_defaults():
    """make_policy('netdmcts') uses sensible defaults."""
    from locma.policies.net_oracle import NetGuidedDMCTSBattlePolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    p = make_policy("netdmcts")
    assert isinstance(p.battle, NetGuidedDMCTSBattlePolicy)
    assert p.battle.K == 15
    assert p.battle.iterations == 80
    assert p.battle.model_path == "model.zip"


def test_netdmcts_registry_bad_spec_raises():
    """An unknown spec raises ValueError."""
    from locma.policies.registry import make_policy  # noqa: PLC0415

    with pytest.raises(ValueError):
        make_policy("netdmcts_typo")


def test_netdmcts_in_policy_names():
    """netdmcts appears in policy_names()."""
    from locma.policies.registry import policy_names  # noqa: PLC0415

    assert "netdmcts" in policy_names()
