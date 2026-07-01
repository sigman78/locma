"""League (fictitious self-play) loop for the token PPO net.

Each round trains the current net against a per-episode mix of all past frozen
snapshots plus the ground baselines, then snapshots itself into the pool. ML
imports are lazy so the pure helpers stay import-safe without the [ml] extra.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

DEFAULT_BASELINES: tuple[str, ...] = ("scripted", "max-guard", "max-attack")


def league_pool_specs(snapshots, baselines=DEFAULT_BASELINES) -> list[str]:
    """FSP pool: every past snapshot as ``ppo:<path>``, then the baselines."""
    return [f"ppo:{s}" for s in snapshots] + list(baselines)


def write_league_csv(path, rows) -> None:
    """Write the per-round tracking CSV, rewritten after every round."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_league_opponent(snapshots, baselines=DEFAULT_BASELINES, seed=0):
    """Build the FSP opponent as a Python object.

    Constructed directly rather than via a spec string so snapshot policies stay
    in a single process and do not need to cross a subprocess pickle boundary.
    """
    from locma.policies.mixed import MixedOpponentPolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    pool = [make_policy(s) for s in league_pool_specs(snapshots, baselines)]
    return MixedOpponentPolicy(pool, name="league", seed=seed)


def _league_env(opponent, seed, obs_mode="token", both_seat=True):
    """Single-env DummyVecEnv wrapping BattleEnv with a direct opponent object."""
    from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: PLC0415

    from locma.envs.battle_env import BattleEnv  # noqa: PLC0415

    fn = functools.partial(
        BattleEnv,
        opponent=opponent,
        seed=seed,
        agent_seat=0,
        seat_random=both_seat,
        obs_mode=obs_mode,
    )
    return DummyVecEnv([fn])


def run_league(
    base_path,
    rounds: int = 6,
    steps_per_round: int = 200_000,
    out_dir: str = "runs/league",
    seed: int = 0,
    baselines=DEFAULT_BASELINES,
    eval_seeds: int = 150,
    eval_base_seed: int = 1_000_000,
    games_per_seed: int = 2,
    target_kl: float = 0.025,
    obs_mode: str = "token",
    verbose: int = 0,
) -> list[dict]:
    """Run a growing-pool FSP league and track avg-hard3 per round."""
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.harness.ar_study import hard3_per_seed  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_list = [eval_base_seed + i for i in range(eval_seeds)]

    def _eval(path) -> float:
        return float(hard3_per_seed(str(path), eval_list, games_per_seed).mean())

    snapshots = [str(base_path)]
    rows = [
        {
            "round": 0,
            "snapshot": str(base_path),
            "avg_hard3": _eval(base_path),
            "n_seeds": eval_seeds,
        }
    ]
    write_league_csv(out / "league.csv", rows)
    if verbose:
        print(f"[league] round 0 (base) avg_hard3={rows[0]['avg_hard3']:.4f}")

    model = MaskablePPO.load(base_path)
    model.target_kl = target_kl
    for r in range(1, rounds + 1):
        opp = build_league_opponent(snapshots, baselines, seed=seed + r)
        env = _league_env(opp, seed + r, obs_mode=obs_mode)
        model.set_env(env)
        model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
        snap = str(out / f"round{r}.zip")
        model.save(snap)
        env.close()

        snapshots.append(snap)
        rows.append(
            {
                "round": r,
                "snapshot": snap,
                "avg_hard3": _eval(snap),
                "n_seeds": eval_seeds,
            }
        )
        write_league_csv(out / "league.csv", rows)
        if verbose:
            print(f"[league] round {r} avg_hard3={rows[-1]['avg_hard3']:.4f}")
    return rows
