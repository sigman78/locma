from __future__ import annotations

import argparse

from sb3_contrib import MaskablePPO

from locma.envs.battle_env import BattleEnv
from locma.policies.random_policy import RandomPolicy


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a MaskablePPO agent on LOCM 1.2.")
    ap.add_argument("--steps", type=int, default=50_000, help="Total training timesteps")
    ap.add_argument("--out", default="model.zip", help="Output path for saved model")
    args = ap.parse_args()

    env = BattleEnv(opponent=RandomPolicy("opp"), seed=0)
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.steps)
    model.save(args.out)
    env.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
