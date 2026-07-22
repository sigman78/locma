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
    args = ap.parse_args()

    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415
    from locma.envs.training import _build_env  # noqa: PLC0415

    opp_device = os.environ.get("LOCMA_PFSP_OPP_DEVICE", "auto")  # matches pfsp default
    env = _build_env(
        f"pfsp:{POOL}",
        args.seed,
        args.n_envs,
        both_seat=True,
        obs_mode="token-fx",
        draft_override=LDRAFT,
    )
    model = MaskablePPO.load(resolve_path(args.warm), env=env, device="auto")

    # warmup: forces the SubprocVecEnv workers to lazily load every pool net,
    # so the timed segment measures steady-state rollout+update throughput.
    model.learn(total_timesteps=args.warmup, reset_num_timesteps=True)

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
    dt = time.perf_counter() - t0
    env.close()

    fps = args.steps / dt if dt > 0 else 0.0
    print(
        json.dumps(
            {
                "n_envs": args.n_envs,
                "opp_device": opp_device,
                "learner_device": str(model.device),
                "timed_steps": args.steps,
                "seconds": round(dt, 2),
                "fps": round(fps, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
