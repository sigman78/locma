"""Gate-0 throughput spike: sb3 SubprocVecEnv vs PufferLib vectorization on BattleEnv.

Throughput only (no learning). Informs a FUTURE PufferLib migration decision; the
ceiling study runs on sb3 regardless. Uses time.perf_counter (allowed) — never call
this from a workflow script (Date/random restrictions there do not apply to scripts).
"""

from __future__ import annotations

import time

import numpy as np


def sb3_sps(n_envs: int, steps: int, obs_mode: str = "token", opponent: str = "scripted") -> float:
    from locma.envs.training import _build_env  # noqa: PLC0415

    vec = _build_env(opponent, seed=0, n_envs=n_envs, both_seat=True, obs_mode=obs_mode)
    vec.reset()
    t0 = time.perf_counter()
    n = 0
    while n < steps:
        actions = np.array([vec.action_space.sample() for _ in range(n_envs)])
        vec.step(actions)
        n += n_envs
    dt = time.perf_counter() - t0
    vec.close()
    return n / dt if dt > 0 else 0.0


def puffer_sps(n_envs: int, steps: int, obs_mode: str = "token", opponent: str = "scripted"):
    try:
        import pufferlib  # noqa: F401, PLC0415
    except ImportError:
        return None
    # PufferLib wiring is intentionally deferred to the run session on the GPU box,
    # where pufferlib is installed; this stub returns None until then so the sb3
    # number is always available. See the runbook (Gate 0).
    return None


def main() -> None:
    print(f"{'config':<22}{'SPS':>12}")
    for ne in (1, 4, 8, 16):
        print(f"sb3 token n_envs={ne:<3}{sb3_sps(ne, 4000):>12.0f}")
    p = puffer_sps(8, 4000)
    print(f"puffer token n_envs=8 {('(absent)' if p is None else f'{p:.0f}'):>12}")


if __name__ == "__main__":
    main()
