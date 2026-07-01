import numpy as np
import pytest

torch = pytest.importorskip("torch")

from locma.envs.action_factor import ACTION_SIZE, decode, factor_masks  # noqa: E402
from locma.envs.ar_distribution import (  # noqa: E402
    decode_batch,
    encode_batch,
    factor_grids,
)


def test_decode_batch_matches_scalar():
    flat = torch.arange(ACTION_SIZE, dtype=torch.long)
    t, s, tgt = decode_batch(flat)
    for idx in range(ACTION_SIZE):
        assert (int(t[idx]), int(s[idx]), int(tgt[idx])) == decode(idx)


def test_encode_batch_inverts_decode():
    flat = torch.arange(ACTION_SIZE, dtype=torch.long)
    t, s, tgt = decode_batch(flat)
    assert torch.equal(encode_batch(t, s, tgt), flat)


def test_factor_grids_match_numpy_factor_masks():
    rng = np.random.default_rng(1)
    flat = rng.random((5, ACTION_SIZE)) < 0.2
    grids = factor_grids(torch.as_tensor(flat))
    for b in range(5):
        tm, sm, tgtm = factor_masks(flat[b])
        assert torch.equal(grids[b].any(dim=1).any(dim=1), torch.as_tensor(tm))
        assert torch.equal(grids[b].any(dim=2), torch.as_tensor(sm))
        assert torch.equal(grids[b], torch.as_tensor(tgtm))
