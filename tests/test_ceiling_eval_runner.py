import pytest

pytest.importorskip("sb3_contrib")

from locma.envs.training import _build_env, _make_model
from locma.harness.ceiling_eval import avg_hard3_per_seed, run_verdict


def _tiny_model(tmp_path, name):
    env = _build_env("random", seed=0, n_envs=1, both_seat=True, obs_mode="flat")
    m = _make_model(env, obs_mode="flat", seed=0, verbose=0, ent_coef=0.02)
    m.learn(total_timesteps=400)
    p = str(tmp_path / f"{name}.zip")
    m.save(p)
    return p


def test_runner_smoke_returns_verdict(tmp_path):
    cand = _tiny_model(tmp_path, "cand")
    b0 = _tiny_model(tmp_path, "b0")
    seeds = [1_000_000, 1_000_001]
    rates = avg_hard3_per_seed(cand, seeds, games_per_seed=2)
    assert len(rates) == len(seeds) and all(0.0 <= r <= 1.0 for r in rates)
    out = run_verdict([cand], [b0], seeds=seeds, games_per_seed=2)
    assert out["verdict"] in ("headroom", "ceiling-confirmed")
    assert set(out) >= {"mean_delta", "ci_lo", "ci_hi", "cand_avg", "b0_avg"}
