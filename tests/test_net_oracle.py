"""Tests for NetOracle: masked policy priors + value from token net.

Requires the [ml] extra (sb3_contrib, torch). All tests are gated with
``pytest.importorskip("sb3_contrib")`` so they are skipped cleanly in CI
without the extra.
"""

from __future__ import annotations

import math
import random

import pytest

from locma.core import battle as battlemod
from locma.core.draft import apply_draft_pick, start_draft
from locma.core.state import GameState, Phase
from locma.data.cards_db import load_cards


def _battle_state(seed: int = 0) -> GameState:
    """Return a GameState at a battle decision point with >1 legal action."""
    # seed=0 gives 3 legal actions at battle start (confirmed via probe)
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


def test_net_oracle_priors_len_sum(tmp_path):
    """priors() returns one prior per legal action, all >= 0, summing to 1."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetOracle  # noqa: PLC0415

    path = _make_token_model(tmp_path)
    oracle = NetOracle(path)

    gs = _battle_state(seed=0)
    legal = list(battlemod.battle_legal(gs))
    assert len(legal) > 1, "test needs >1 legal action"

    p = oracle.priors(gs, legal, gs.current)

    assert len(p) == len(legal), f"priors length {len(p)} != legal {len(legal)}"
    assert all(x >= 0.0 for x in p), f"negative prior: {p}"
    assert abs(sum(p) - 1.0) < 1e-5, f"priors sum {sum(p):.8f} != 1.0"


def test_net_oracle_value_in_range(tmp_path):
    """value() returns a finite float in [-1.0, 1.0]."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetOracle  # noqa: PLC0415

    path = _make_token_model(tmp_path)
    oracle = NetOracle(path)

    gs = _battle_state(seed=0)
    v = oracle.value(gs, gs.current)

    assert math.isfinite(v), f"value not finite: {v}"
    assert -1.0 <= v <= 1.0, f"value {v} outside [-1, 1]"


def test_net_oracle_value_sign_flip(tmp_path):
    """value(gs, 1-seat) == -value(gs, seat) when |v| < 1."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetOracle  # noqa: PLC0415

    path = _make_token_model(tmp_path)
    oracle = NetOracle(path)

    # seed=0 raw value is ~-0.265, well within (-1, 1)
    gs = _battle_state(seed=0)
    seat = gs.current

    v_me = oracle.value(gs, seat)
    v_op = oracle.value(gs, 1 - seat)

    assert math.isfinite(v_me) and math.isfinite(v_op)
    # Both clipped; |v| < 1 so negation holds exactly within fp tolerance
    assert abs(v_me + v_op) < 1e-6, f"sign-flip failed: v_me={v_me}, v_op={v_op}"


def test_net_oracle_lazy_load(tmp_path):
    """NetOracle does not load the model at construction time."""
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetOracle  # noqa: PLC0415

    oracle = NetOracle("nonexistent.zip")
    assert oracle._model is None
