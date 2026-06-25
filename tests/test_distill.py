# tests/test_distill.py
import json

import numpy as np
import pytest

from locma.envs.distill import load_practicum, split_by_game
from locma.envs.encode import ACTION_SIZE, OBS_SIZE
from locma.envs.practicum import _manifest_path


def _write_fake_practicum(path, n=20, n_games=5):
    rng = np.random.default_rng(0)
    game_id = np.repeat(np.arange(n_games, dtype=np.int32), n // n_games)
    np.savez(
        path,
        obs=rng.standard_normal((n, OBS_SIZE)).astype(np.float32),
        action=rng.integers(0, ACTION_SIZE, size=n).astype(np.int64),
        mask=np.ones((n, ACTION_SIZE), dtype=bool),
        winner=np.zeros(n, dtype=np.int8),
        seat=np.zeros(n, dtype=np.int8),
        opponent_id=np.zeros(n, dtype=np.int8),
        game_id=game_id,
    )
    with open(_manifest_path(str(path)), "w") as f:
        json.dump({"obs_size": OBS_SIZE, "action_size": ACTION_SIZE}, f)


def test_split_by_game_has_no_leakage(tmp_path):
    game_id = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    train_idx, val_idx = split_by_game(game_id, val_frac=0.5, seed=0)
    train_games = {int(game_id[i]) for i in train_idx}
    val_games = {int(game_id[i]) for i in val_idx}
    assert train_games.isdisjoint(val_games)
    assert train_games | val_games == {0, 1, 2, 3}
    assert set(train_idx) | set(val_idx) == set(range(8))


def test_load_practicum_round_trips(tmp_path):
    p = tmp_path / "practicum.npz"
    _write_fake_practicum(p)
    arrays, manifest = load_practicum(str(p))
    assert arrays["obs"].shape == (20, OBS_SIZE)
    assert manifest["obs_size"] == OBS_SIZE


def test_load_practicum_rejects_stale_layout(tmp_path):
    p = tmp_path / "practicum.npz"
    _write_fake_practicum(p)
    with open(_manifest_path(str(p)), "w") as f:
        json.dump({"obs_size": OBS_SIZE + 1, "action_size": ACTION_SIZE}, f)
    with pytest.raises(ValueError, match="layout"):
        load_practicum(str(p))
