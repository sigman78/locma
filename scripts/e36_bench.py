"""E36 throughput micro-benchmark: PFSP best-response FPS vs n_envs / opp device.

Runs ONE config (n_envs + opponent device via LOCMA_PFSP_OPP_DEVICE), warms up so
the lazy pool nets are loaded, then times a fixed step budget and prints a JSON
line with steps-per-second. Invoke it several times (see scripts/e36_bench.sh) to
A/B configs in fresh processes — avoids GPU-memory / SubprocVecEnv carryover.

    LOCMA_PFSP_OPP_DEVICE=cpu .venv/Scripts/python scripts/e36_bench.py --n-envs 12
"""

from __future__ import annotations

import argparse
import json
import os
import time

E29 = "depot:e29slim/e29slim_s0.zip"
LDRAFT = "depot:ldraft/ldraft_s0.zip"
POOL = "runs/e36/pool.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=8192, help="untimed steps to load pool nets")
    ap.add_argument("--steps", type=int, default=60000, help="timed step budget")
    ap.add_argument("--warm", default=E29)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--driver", choices=["subproc", "batched"], default="subproc")
    ap.add_argument("--device", default="auto", help="SB3 learner device: auto|cpu|cuda|mps")
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415

    opp_device = os.environ.get("LOCMA_PFSP_OPP_DEVICE", "auto")  # matches pfsp default
    if args.driver == "batched":
        from locma.envs.batched_selfplay import make_batched_opponent_vecenv  # noqa: PLC0415

        env = make_batched_opponent_vecenv(
            POOL, args.n_envs, seed=args.seed, ldraft=LDRAFT, obs_variant="fx"
        )
    else:
        from locma.envs.training import _build_env  # noqa: PLC0415

        env = _build_env(
            f"pfsp:{POOL}",
            args.seed,
            args.n_envs,
            both_seat=True,
            obs_mode="token-fx",
            draft_override=LDRAFT,
        )
    model = MaskablePPO.load(resolve_path(args.warm), env=env, device=args.device)
    model.verbose = 0  # silence SB3's per-iteration fps table (ambiguous vs the timed number below)

    # warmup: forces the SubprocVecEnv workers to lazily load every pool net,
    # so the timed segment measures steady-state rollout+update throughput.
    model.learn(total_timesteps=args.warmup, reset_num_timesteps=True)

    # Measure over the ACTUAL steps SB3 ran, not the requested budget: learn()
    # always collects whole rollouts (n_steps*n_envs) and rounds the budget UP, so
    # dividing by args.steps mis-reports (and the skew depends on n_envs). This is
    # true end-to-end throughput: rollout collection + PPO update (incl. KL stop).
    n0 = model.num_timesteps
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
    dt = time.perf_counter() - t0
    actual_steps = model.num_timesteps - n0
    env.close()

    fps = actual_steps / dt if dt > 0 else 0.0
    print(
        json.dumps(
            {
                "driver": args.driver,
                "n_envs": args.n_envs,
                "opp_device": opp_device,
                "learner_device": str(model.device),
                "actual_steps": actual_steps,
                "seconds": round(dt, 2),
                "fps": round(fps, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
