import numpy as np

from locma.envs.action_factor import (
    ACTION_SIZE,
    ATTACK,
    PASS,
    SUMMON,
    USE,
    decode,
    encode,
    factor_masks,
)


def test_round_trip_all_indices():
    for idx in range(ACTION_SIZE):
        t, s, tgt = decode(idx)
        assert encode(t, s, tgt) == idx


def test_decode_boundaries():
    assert decode(0) == (PASS, 0, 0)
    assert decode(1) == (SUMMON, 0, 0)
    assert decode(8) == (SUMMON, 7, 0)
    assert decode(9) == (USE, 0, 0)
    assert decode(112) == (USE, 7, 12)
    assert decode(113) == (ATTACK, 0, 0)
    assert decode(154) == (ATTACK, 5, 6)


def test_factor_masks_reconstruct_flat():
    rng = np.random.default_rng(0)
    for _ in range(200):
        flat = rng.random(ACTION_SIZE) < 0.15
        tm, sm, tgtm = factor_masks(flat)
        rebuilt = np.zeros(ACTION_SIZE, dtype=bool)
        for idx in range(ACTION_SIZE):
            t, s, tgt = decode(idx)
            if tm[t] and sm[t, s] and tgtm[t, s, tgt]:
                rebuilt[idx] = True
        assert np.array_equal(rebuilt, flat)


def test_factor_masks_nonapplicable_single_cell():
    flat = np.zeros(ACTION_SIZE, dtype=bool)
    flat[0] = True  # Pass legal
    flat[1] = True  # Summon slot 0 legal
    tm, sm, tgtm = factor_masks(flat)
    assert sm[PASS].sum() == 1 and tgtm[PASS, 0].sum() == 1
    assert tgtm[SUMMON, 0].sum() == 1
