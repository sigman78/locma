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


from locma.envs.ar_distribution import ARHeads, ar_evaluate, ar_sample  # noqa: E402


def _masks_batch(seed, b=16):
    rng = np.random.default_rng(seed)
    flats = np.zeros((b, ACTION_SIZE), dtype=bool)
    for i in range(b):
        flats[i, 0] = True  # Pass always legal
        extra = rng.integers(1, ACTION_SIZE, size=rng.integers(1, 6))
        flats[i, extra] = True
    return torch.as_tensor(flats)


def test_sampled_actions_are_always_legal():
    torch.manual_seed(0)
    heads = ARHeads(latent_dim=12)
    flat_masks = _masks_batch(2)
    z = torch.randn(flat_masks.shape[0], 12)
    actions, _ = ar_sample(heads, z, flat_masks, deterministic=False)
    for i, a in enumerate(actions.tolist()):
        assert flat_masks[i, a]


def test_log_prob_equals_sum_of_conditionals_and_finite():
    torch.manual_seed(1)
    heads = ARHeads(latent_dim=12)
    flat_masks = _masks_batch(3)
    z = torch.randn(flat_masks.shape[0], 12)
    actions, lp_sample = ar_sample(heads, z, flat_masks, deterministic=True)
    lp_eval, ent = ar_evaluate(heads, z, flat_masks, actions)
    assert torch.allclose(lp_sample, lp_eval, atol=1e-5)
    assert torch.isfinite(lp_eval).all()
    assert torch.isfinite(ent).all()
    assert (ent >= -1e-6).all()


def test_pass_only_has_zero_logprob_and_entropy():
    torch.manual_seed(2)
    heads = ARHeads(latent_dim=12)
    flat = np.zeros((1, ACTION_SIZE), dtype=bool)
    flat[0, 0] = True  # only Pass legal
    flat_masks = torch.as_tensor(flat)
    z = torch.randn(1, 12)
    actions, lp = ar_sample(heads, z, flat_masks, deterministic=True)
    assert int(actions[0]) == 0
    assert torch.allclose(lp, torch.zeros_like(lp), atol=1e-5)
    _, ent = ar_evaluate(heads, z, flat_masks, actions)
    assert torch.allclose(ent, torch.zeros_like(ent), atol=1e-5)
