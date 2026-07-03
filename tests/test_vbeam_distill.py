"""Tests for distilling the vbeam planner into the reactive net (E4v2 / EXIT).

Collection reuses collect_value_data with a fast heuristic teacher so no model
artifact is needed; training tests are gated on the [ml] extra.
"""

from __future__ import annotations

import numpy as np
import pytest

from locma.envs.vbeam_fvi import collect_value_data


def _make_token_model(tmp_path):
    pytest.importorskip("sb3_contrib")
    from locma.envs.training import _build_env, _make_model  # noqa: PLC0415

    env = _build_env("random", 0, 1, obs_mode="token")
    model = _make_model(env, obs_mode="token", seed=0, verbose=0, ent_coef=0.02)
    path = str(tmp_path / "base.zip")
    model.save(path)
    env.close()
    return path


def test_train_policy_head_freezes_critic_path(tmp_path):
    """The load-bearing invariant, mirrored from FVI: only the policy branch
    may change — the extractor and the whole critic path must stay
    byte-identical, because vbeam's plan ranking reads the critic."""
    pytest.importorskip("sb3_contrib")
    import torch  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.envs.vbeam_distill import train_policy_head  # noqa: PLC0415

    base = _make_token_model(tmp_path)
    data = str(tmp_path / "data.npz")
    collect_value_data("greedy", data, opponents=("random",), games=4, seed=0, workers=1)

    out = str(tmp_path / "ph.zip")
    metrics = train_policy_head(base, data, out, epochs=3, batch_size=64, seed=0)

    ref = MaskablePPO.load(base).policy.state_dict()
    new = MaskablePPO.load(out).policy.state_dict()
    assert ref.keys() == new.keys()

    changed, frozen_ok = [], True
    for k in ref:
        same = torch.equal(ref[k], new[k])
        is_pi = "mlp_extractor.policy_net" in k or k.startswith("action_net")
        if is_pi and not same:
            changed.append(k)
        if not is_pi and not same:
            frozen_ok = False
    assert frozen_ok, "a non-policy parameter changed"
    assert changed, "no policy parameter changed — training was a no-op"

    assert metrics["n_examples"] > 0
    assert np.isfinite(metrics["val_ce_before"]) and np.isfinite(metrics["val_ce_after"])
    assert 0.0 <= metrics["val_agreement_after"] <= 1.0


def test_train_policy_head_learns_the_data(tmp_path):
    """CE toward the teacher actions goes down on train data (sanity that the
    masked-logit path carries gradient)."""
    pytest.importorskip("sb3_contrib")
    from locma.envs.vbeam_distill import train_policy_head  # noqa: PLC0415

    base = _make_token_model(tmp_path)
    data = str(tmp_path / "data.npz")
    collect_value_data("greedy", data, opponents=("random",), games=4, seed=1, workers=1)

    out = str(tmp_path / "ph.zip")
    metrics = train_policy_head(base, data, out, epochs=5, batch_size=64, seed=0)
    assert metrics["val_ce_after"] < metrics["val_ce_before"]


def test_trained_model_is_vbeam_drop_in_with_identical_values(tmp_path):
    """The PH output loads in NetValueEvaluator and its critic values match the
    base model exactly (the frozen-critic promise, observed end to end)."""
    pytest.importorskip("sb3_contrib")
    import random  # noqa: PLC0415

    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.draft import apply_draft_pick, start_draft  # noqa: PLC0415
    from locma.core.engine import make_battle_view  # noqa: PLC0415
    from locma.core.state import GameState, Phase  # noqa: PLC0415
    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.envs.vbeam_distill import train_policy_head  # noqa: PLC0415
    from locma.policies.vbeam import NetValueEvaluator  # noqa: PLC0415

    base = _make_token_model(tmp_path)
    data = str(tmp_path / "data.npz")
    collect_value_data("greedy", data, opponents=("random",), games=3, seed=0, workers=1)
    out = str(tmp_path / "ph.zip")
    train_policy_head(base, data, out, epochs=2, batch_size=64, seed=0)

    gs = GameState.new(random.Random(0))
    start_draft(gs, load_cards())
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
    battlemod.start_battle(gs)
    view = make_battle_view(gs)

    v_base = NetValueEvaluator(base).values([view])
    v_ph = NetValueEvaluator(out).values([view])
    assert v_base == pytest.approx(v_ph, abs=1e-6)


def test_behavior_clone_init_model_warm_start(tmp_path):
    """init_model warm-starts BC from a saved token net; a fresh-vs-warm run
    differ (the warm start is real) and both save loadable models."""
    pytest.importorskip("sb3_contrib")
    import torch  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.envs.distill import behavior_clone  # noqa: PLC0415

    base = _make_token_model(tmp_path)
    data = str(tmp_path / "data.npz")
    collect_value_data("greedy", data, opponents=("random",), games=3, seed=0, workers=1)

    warm = str(tmp_path / "warm.zip")
    res = behavior_clone(
        data=data, out=warm, epochs=1, batch=64, seed=0, verbose=0, init_model=base
    )
    assert np.isfinite(res["val_agreement"]) or res["n_val"] == 0

    # Warm start really started from base: the (untouched-by-CE) value head
    # still matches base, while the policy head moved.
    ref = MaskablePPO.load(base).policy.state_dict()
    new = MaskablePPO.load(warm).policy.state_dict()
    assert torch.equal(ref["value_net.weight"], new["value_net.weight"])
    assert not torch.equal(ref["action_net.weight"], new["action_net.weight"])


def test_behavior_clone_init_model_obs_mismatch(tmp_path):
    """A flat init_model against a token practicum fails loudly."""
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO  # noqa: PLC0415
    from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: PLC0415

    from locma.envs.distill import behavior_clone  # noqa: PLC0415
    from locma.envs.training import _make_battle_env  # noqa: PLC0415

    env = DummyVecEnv([lambda: _make_battle_env("random", 0)])
    flat = str(tmp_path / "flat.zip")
    MaskablePPO("MlpPolicy", env, seed=0, verbose=0).save(flat)
    env.close()

    data = str(tmp_path / "data.npz")
    collect_value_data("greedy", data, opponents=("random",), games=2, seed=0, workers=1)

    with pytest.raises(ValueError, match="init_model"):
        behavior_clone(data=data, out=str(tmp_path / "x.zip"), epochs=1, init_model=flat)
