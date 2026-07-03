"""Tests for fitted-value iteration on the vbeam critic (E5 variant 2).

Collection tests use a fast heuristic teacher (greedy) so no model artifact is
needed; training tests are gated on the [ml] extra.
"""

from __future__ import annotations

import numpy as np
import pytest

from locma.envs.vbeam_fvi import _value_targets, collect_value_data


def test_collect_sharded_matches_serial(tmp_path):
    """workers=2 with one opponent merges to byte-identical arrays vs workers=1."""
    serial = str(tmp_path / "serial.npz")
    sharded = str(tmp_path / "sharded.npz")
    m1 = collect_value_data("greedy", serial, opponents=("random",), games=4, seed=7, workers=1)
    m2 = collect_value_data("greedy", sharded, opponents=("random",), games=4, seed=7, workers=2)

    assert m1["n_examples"] == m2["n_examples"] > 0
    with np.load(serial) as a, np.load(sharded) as b:
        for k in ("obs_tokens", "obs_scalars", "action", "winner", "seat", "game_id"):
            assert np.array_equal(a[k], b[k]), f"array {k} differs between serial and sharded"


def test_collect_cleans_up_shards(tmp_path):
    out = str(tmp_path / "data.npz")
    collect_value_data("greedy", out, opponents=("random",), games=2, seed=0, workers=2)
    leftovers = [p.name for p in tmp_path.iterdir() if "shard" in p.name]
    assert leftovers == []


def test_value_targets_sign():
    data = {"winner": np.array([0, 1, 1, 0]), "seat": np.array([0, 0, 1, 1])}
    assert _value_targets(data).tolist() == [1.0, -1.0, 1.0, -1.0]


def test_value_targets_prefers_explicit_target_column():
    data = {
        "winner": np.array([0, 1]),
        "seat": np.array([0, 0]),
        "target": np.array([0.25, -0.5], dtype=np.float32),
    }
    assert _value_targets(data).tolist() == [0.25, -0.5]


# ---------------------------------------------------------------------------
# train_value_head — [ml]-gated
# ---------------------------------------------------------------------------


def _make_token_model(tmp_path):
    pytest.importorskip("sb3_contrib")
    from locma.envs.training import _build_env, _make_model  # noqa: PLC0415

    env = _build_env("random", 0, 1, obs_mode="token")
    model = _make_model(env, obs_mode="token", seed=0, verbose=0, ent_coef=0.02)
    path = str(tmp_path / "base.zip")
    model.save(path)
    env.close()
    return path


def test_train_value_head_freezes_policy_path(tmp_path):
    """The load-bearing invariant: only the critic branch may change — the
    policy path (extractor, pi MLP, action head) must stay byte-identical,
    because vbeam's stop rule reads the policy head's masked argmax."""
    pytest.importorskip("sb3_contrib")
    import torch  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.envs.vbeam_fvi import train_value_head  # noqa: PLC0415

    base = _make_token_model(tmp_path)
    data = str(tmp_path / "data.npz")
    collect_value_data("greedy", data, opponents=("random",), games=3, seed=0, workers=1)

    out = str(tmp_path / "fvi.zip")
    metrics = train_value_head(base, data, out, epochs=3, batch_size=64, seed=0)

    ref = MaskablePPO.load(base).policy.state_dict()
    new = MaskablePPO.load(out).policy.state_dict()
    assert ref.keys() == new.keys()

    changed, frozen_ok = [], True
    for k in ref:
        same = torch.equal(ref[k], new[k])
        is_critic = "mlp_extractor.value_net" in k or k.startswith("value_net")
        if is_critic and not same:
            changed.append(k)
        if not is_critic and not same:
            frozen_ok = False
    assert frozen_ok, "a non-critic parameter changed"
    assert changed, "no critic parameter changed — training was a no-op"

    assert metrics["n_examples"] > 0
    assert np.isfinite(metrics["val_mse_before"]) and np.isfinite(metrics["val_mse_after"])


def test_collect_backup_data_end_to_end(tmp_path):
    """Backed-up-target collection writes a target column that train_value_head
    consumes; the pi-frozen invariant holds on that path too."""
    pytest.importorskip("sb3_contrib")
    import torch  # noqa: PLC0415
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.envs.vbeam_fvi import collect_backup_data, train_value_head  # noqa: PLC0415

    base = _make_token_model(tmp_path)
    data = str(tmp_path / "backup.npz")
    m = collect_backup_data(
        base, data, opponents=("random",), games=2, seed=0, workers=1, width=4, max_actions=10
    )
    assert m["n_examples"] > 0
    with np.load(data) as d:
        assert "target" in d.files
        assert d["target"].min() >= -1.0 and d["target"].max() <= 1.0
        assert len(d["target"]) == len(d["obs_tokens"]) == m["n_examples"]

    out = str(tmp_path / "fvi-az.zip")
    metrics = train_value_head(base, data, out, epochs=2, batch_size=64, seed=0)
    assert np.isfinite(metrics["val_mse_after"])

    ref = MaskablePPO.load(base).policy.state_dict()
    new = MaskablePPO.load(out).policy.state_dict()
    for k in ref:
        if "mlp_extractor.value_net" in k or k.startswith("value_net"):
            continue
        assert torch.equal(ref[k], new[k]), f"non-critic parameter changed: {k}"


def test_trained_model_is_vbeam_drop_in(tmp_path):
    """The FVI output loads in NetValueEvaluator and yields finite values."""
    pytest.importorskip("sb3_contrib")
    import random  # noqa: PLC0415

    from locma.core import battle as battlemod  # noqa: PLC0415
    from locma.core.draft import apply_draft_pick, start_draft  # noqa: PLC0415
    from locma.core.engine import make_battle_view  # noqa: PLC0415
    from locma.core.state import GameState, Phase  # noqa: PLC0415
    from locma.data.cards_db import load_cards  # noqa: PLC0415
    from locma.envs.encode import action_mask  # noqa: PLC0415
    from locma.envs.vbeam_fvi import train_value_head  # noqa: PLC0415
    from locma.policies.vbeam import NetValueEvaluator  # noqa: PLC0415

    base = _make_token_model(tmp_path)
    data = str(tmp_path / "data.npz")
    collect_value_data("greedy", data, opponents=("random",), games=2, seed=0, workers=1)
    out = str(tmp_path / "fvi.zip")
    train_value_head(base, data, out, epochs=2, batch_size=64, seed=0)

    gs = GameState.new(random.Random(0))
    start_draft(gs, load_cards())
    while gs.phase == Phase.DRAFT:
        apply_draft_pick(gs, 0)
    battlemod.start_battle(gs)
    view = make_battle_view(gs)
    mask = action_mask(view, battlemod.battle_legal(gs))

    vals, would_pass = NetValueEvaluator(out).evaluate([view], [mask])
    assert -1.0 <= vals[0] <= 1.0
    assert isinstance(would_pass[0], bool)
