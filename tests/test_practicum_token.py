"""Tests for token-observation mode in practicum recording.

No [ml] extras required: uses only numpy + heuristic teachers.
"""

import json

import numpy as np
import pytest

from locma.envs.encode import ACTION_SIZE, MAX_TOKENS, N_TACTICAL, OBS_SIZE, TOKEN_FEATS
from locma.envs.practicum import _Collector, _manifest_path, record_practicum

# ---------------------------------------------------------------------------
# 1. Token-mode integration: real run, correct arrays + manifest
# ---------------------------------------------------------------------------


def test_record_practicum_token_writes_four_obs_arrays(tmp_path):
    out = str(tmp_path / "p.npz")
    manifest = record_practicum(
        teacher="greedy",
        opponents=("random",),
        games=1,
        out=out,
        seed=0,
        obs_mode="token",
    )
    n = manifest["n_examples"]
    assert n > 0, "expected at least one recorded example"

    with np.load(out) as d:
        # four token arrays must exist
        assert "obs_tokens" in d, "obs_tokens key missing"
        assert "obs_card_ids" in d, "obs_card_ids key missing"
        assert "obs_token_mask" in d, "obs_token_mask key missing"
        assert "obs_scalars" in d, "obs_scalars key missing"

        # NO flat obs key in token mode
        assert "obs" not in d, "obs key must NOT be present in token mode"

        # shapes and dtypes
        assert d["obs_tokens"].shape == (n, MAX_TOKENS, TOKEN_FEATS)
        assert d["obs_tokens"].dtype == np.float32
        assert d["obs_card_ids"].shape == (n, MAX_TOKENS)
        assert d["obs_card_ids"].dtype == np.float32
        assert d["obs_token_mask"].shape == (n, MAX_TOKENS)
        assert d["obs_token_mask"].dtype == np.float32
        assert d["obs_scalars"].shape == (n, N_TACTICAL)
        assert d["obs_scalars"].dtype == np.float32

        # unchanged non-obs arrays
        assert d["action"].shape == (n,)
        assert d["mask"].shape == (n, ACTION_SIZE)
        assert d["mask"].dtype == bool


def test_record_practicum_token_manifest(tmp_path):
    out = str(tmp_path / "p.npz")
    record_practicum(
        teacher="greedy",
        opponents=("random",),
        games=1,
        out=out,
        seed=0,
        obs_mode="token",
    )
    with open(_manifest_path(out)) as f:
        m = json.load(f)

    assert m["obs_mode"] == "token"
    assert m["max_tokens"] == MAX_TOKENS  # 20
    assert m["token_feats"] == TOKEN_FEATS  # 17
    assert m["n_tactical"] == N_TACTICAL  # 13
    assert m["action_size"] == ACTION_SIZE  # 155
    # obs_size must NOT be present in token mode
    assert "obs_size" not in m, "obs_size must not appear in token manifest"


# ---------------------------------------------------------------------------
# 2. Flat mode still writes obs (n, OBS_SIZE) and has obs_mode in manifest
# ---------------------------------------------------------------------------


def test_record_practicum_flat_unchanged(tmp_path):
    out = str(tmp_path / "flat.npz")
    manifest = record_practicum(
        teacher="greedy",
        opponents=("random",),
        games=1,
        out=out,
        seed=0,
        obs_mode="flat",
    )
    n = manifest["n_examples"]
    assert n > 0

    with np.load(out) as d:
        assert "obs" in d
        assert d["obs"].shape == (n, OBS_SIZE)
        # token keys must NOT appear
        assert "obs_tokens" not in d

    with open(_manifest_path(out)) as f:
        m = json.load(f)

    assert m["obs_mode"] == "flat"
    assert m["obs_size"] == OBS_SIZE


# ---------------------------------------------------------------------------
# 3. Default obs_mode is flat (backward-compatible)
# ---------------------------------------------------------------------------


def test_record_practicum_default_is_flat(tmp_path):
    out = str(tmp_path / "default.npz")
    record_practicum(
        teacher="greedy",
        opponents=("random",),
        games=1,
        out=out,
        seed=0,
        # no obs_mode arg
    )
    with np.load(out) as d:
        assert "obs" in d
        assert "obs_tokens" not in d

    with open(_manifest_path(out)) as f:
        m = json.load(f)
    assert m["obs_mode"] == "flat"


# ---------------------------------------------------------------------------
# 4. Invalid obs_mode raises ValueError
# ---------------------------------------------------------------------------


def test_record_practicum_invalid_obs_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="obs_mode"):
        record_practicum(
            teacher="greedy",
            opponents=("random",),
            games=1,
            out=str(tmp_path / "x.npz"),
            seed=0,
            obs_mode="bogus",
        )


# ---------------------------------------------------------------------------
# 5. _Collector unit test: token mode appends dicts
# ---------------------------------------------------------------------------


def test_collector_token_mode_appends_dict(monkeypatch):
    import locma.envs.practicum as P  # noqa: PLC0415
    from locma.core.actions import Attack, Pass  # noqa: PLC0415

    fake_dict = {
        "tokens": np.zeros((MAX_TOKENS, TOKEN_FEATS), dtype=np.float32),
        "card_ids": np.zeros(MAX_TOKENS, dtype=np.float32),
        "token_mask": np.zeros(MAX_TOKENS, dtype=np.float32),
        "scalars": np.zeros(N_TACTICAL, dtype=np.float32),
    }

    monkeypatch.setattr(P, "battle_legal", lambda gs: [Attack(1, -1), Pass()])
    monkeypatch.setattr(P, "make_battle_view", lambda gs: object())
    monkeypatch.setattr(P, "encode_battle_tokens", lambda view: fake_dict)
    monkeypatch.setattr(P, "action_mask", lambda view, legal: np.zeros(ACTION_SIZE, dtype=bool))
    monkeypatch.setattr(P, "sem_index", lambda view, a: 0)

    c = _Collector(teacher_seat=0, obs_mode="token")
    c(0, Attack(1, -1), object())

    assert len(c.obs) == 1
    assert isinstance(c.obs[0], dict)
    assert set(c.obs[0].keys()) == {"tokens", "card_ids", "token_mask", "scalars"}
