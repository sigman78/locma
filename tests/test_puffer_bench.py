import pytest

pytest.importorskip("sb3_contrib")

from scripts.puffer_bench import sb3_sps


def test_sb3_sps_positive():
    sps = sb3_sps(n_envs=1, steps=200, obs_mode="flat", opponent="random")
    assert sps > 0.0
