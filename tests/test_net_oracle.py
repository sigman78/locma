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


# ---------------------------------------------------------------------------
# P10 optimisation tests
# ---------------------------------------------------------------------------


def test_net_oracle_combined_forward_equivalence(tmp_path):
    """priors() and value() after optimisation are numerically identical to the
    old separate-forward paths (within 1e-5).

    Reference is built directly via the sb3-contrib internal API (the old
    ``get_distribution`` / ``predict_values`` route).  Oracle outputs must match.
    """
    pytest.importorskip("sb3_contrib")
    import torch  # noqa: PLC0415

    from locma.core.engine import make_battle_view  # noqa: PLC0415
    from locma.envs.encode import action_mask, encode_battle_tokens, sem_index  # noqa: PLC0415
    from locma.policies.net_oracle import NetOracle  # noqa: PLC0415

    path = _make_token_model(tmp_path)
    oracle = NetOracle(path)

    gs = _battle_state(seed=0)
    legal = list(battlemod.battle_legal(gs))
    assert len(legal) > 1, "test needs >1 legal action"

    seat = gs.current
    view = make_battle_view(gs)
    obs = encode_battle_tokens(view)

    # Eagerly load the model so we can access the policy directly
    oracle._ensure()
    policy = oracle._model.policy

    obs_t, _ = policy.obs_to_tensor(obs)
    mask = action_mask(view, legal)

    # Reference: old separate-forward approach
    with torch.no_grad():
        dist_ref = policy.get_distribution(obs_t, action_masks=mask)
        probs_ref = dist_ref.distribution.probs[0].cpu().numpy()  # [155]
        val_ref = float(policy.predict_values(obs_t).item())

    # Map reference probs to legal-action vector (same logic as NetOracle.priors)
    collected_ref = []
    for a in legal:
        idx = sem_index(view, a)
        collected_ref.append(float(probs_ref[idx]) if idx is not None else 0.0)
    total_ref = sum(collected_ref)
    priors_ref = [x / total_ref for x in collected_ref]
    value_ref = max(-1.0, min(1.0, val_ref))  # no flip: seat == sim.current

    # Oracle outputs (uses the optimised combined forward after P10)
    priors_oracle = oracle.priors(gs, legal, seat)
    value_oracle = oracle.value(gs, seat)

    assert len(priors_oracle) == len(legal)
    for i, (p_o, p_r) in enumerate(zip(priors_oracle, priors_ref, strict=True)):
        assert abs(p_o - p_r) < 1e-5, (
            f"priors[{i}] diverged: oracle={p_o:.8f} ref={p_r:.8f} diff={abs(p_o - p_r):.2e}"
        )
    assert abs(value_oracle - value_ref) < 1e-5, (
        f"value diverged: oracle={value_oracle:.8f} ref={value_ref:.8f} "
        f"diff={abs(value_oracle - value_ref):.2e}"
    )


def test_net_oracle_cache_reuse(tmp_path):
    """Cache path: value(sim) after priors(sim) reuses the cached raw value.

    Also verifies that value() on a *different* sim (no preceding priors) still
    returns the correct result via the standalone-forward fallback.
    """
    pytest.importorskip("sb3_contrib")
    from locma.policies.net_oracle import NetOracle  # noqa: PLC0415

    path = _make_token_model(tmp_path)
    oracle = NetOracle(path)

    gs = _battle_state(seed=0)
    legal = list(battlemod.battle_legal(gs))
    seat = gs.current

    # --- Cached path ---
    # priors() populates the cache; value() on the SAME object should use it.
    oracle.priors(gs, legal, seat)
    assert oracle._value_cache is not None, "cache not populated after priors()"
    assert oracle._value_cache[0] is gs, "cache sim reference mismatch"

    v_cached = oracle.value(gs, seat)
    assert math.isfinite(v_cached) and -1.0 <= v_cached <= 1.0

    # The cached value must be CORRECT (non-stale): equal to a standalone,
    # cache-bypassed computation on the same gs+seat.  A fresh oracle has no
    # cache, so value() takes the standalone _forward(view, None) path.
    oracle_ref = NetOracle(path)
    v_standalone_same = oracle_ref.value(gs, seat)
    assert abs(v_cached - v_standalone_same) < 1e-5, (
        f"cached value wrong: {v_cached:.8f} vs {v_standalone_same:.8f}"
    )

    # --- Standalone-fallback path ---
    # A fresh oracle (no prior priors() call) on a different sim must still work.
    gs2 = _battle_state(seed=1)
    oracle2 = NetOracle(path)
    v_standalone = oracle2.value(gs2, gs2.current)
    assert math.isfinite(v_standalone) and -1.0 <= v_standalone <= 1.0

    # --- Stale-cache guard ---
    # After priors(gs) the cache holds gs; calling value(gs2) must NOT return
    # the stale cached value — it must fall back to a fresh forward.
    v_gs2_via_oracle1 = oracle.value(gs2, gs2.current)
    assert math.isfinite(v_gs2_via_oracle1) and -1.0 <= v_gs2_via_oracle1 <= 1.0
    # The two standalone results must agree (same oracle, same model, same state)
    assert abs(v_gs2_via_oracle1 - v_standalone) < 1e-5, (
        f"stale-cache fallback mismatch: {v_gs2_via_oracle1:.8f} vs {v_standalone:.8f}"
    )
