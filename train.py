from __future__ import annotations

import argparse

from locma.envs.training import train_agent
from locma.policies.random_policy import RandomPolicy


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a MaskablePPO agent on LOCM 1.2.")
    ap.add_argument("--steps", type=int, default=50_000, help="Total training timesteps")
    ap.add_argument("--out", default="model.zip", help="Output path for saved model")
    args = ap.parse_args()

    out = train_agent(RandomPolicy("opp"), steps=args.steps, out=args.out, seed=0)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
