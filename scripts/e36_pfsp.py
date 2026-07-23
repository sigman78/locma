"""E36: Prioritized Fictitious Self-Play (PFSP) training loop for the reactive net.

Adapts ByteRL's population fictitious-play (arXiv 2303.04096) to our single-GPU
stack: instead of the fixed scripted zoo, best-respond to a POOL of frozen
opponents sampled per game, prioritised toward the ones we're losing to. The net
is warm-started each generation from the previous best (continued best-response),
keeping the E29 slim + E28 pointer arch and ldraft.

Per generation g:
  1. train (warm-start from the current net) a PPO best-response vs pfsp:pool.json
  2. eval the new net vs each pool member (win rates)
  3. reweight the pool toward losing matchups (PFSP), admit the new net, cap size
  4. rewrite pool.json

Gate 0 = 1-2 generations: does the self-play net beat the start net head-to-head,
hold avg-hard3, and reduce boardkeep exploitability? (Those gate evals are run
separately with the existing tools.) The paper used cluster-scale compute; this
is a scaled-down signal test.

Arch note (from the ByteRL arch comparison): our "capacity/recurrence doesn't
help" verdicts were all measured under the WEAK fixed-zoo signal, so once the
self-play signal is richer they should be RE-TESTED (recurrence esp.) — not
assumed. This driver isolates the training-regime lever first.

Usage:
    .venv/Scripts/python scripts/e36_pfsp.py --generations 1 --steps 3000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

E29 = "depot:e29slim/e29slim_s0.zip"
LDRAFT = "depot:ldraft/ldraft_s0.zip"
# gen-0 seed pool: a frozen strong self + the scripted zoo + the known exploiter.
SEED_POOL = [
    {"spec": f"ppo:{E29},{LDRAFT}", "weight": 2.0, "kind": "self"},
    {"spec": "boardkeep", "weight": 1.0, "kind": "anchor"},
    {"spec": "scripted", "weight": 1.0, "kind": "anchor"},
    {"spec": "max-guard", "weight": 1.0, "kind": "anchor"},
    {"spec": "max-attack", "weight": 1.0, "kind": "anchor"},
]
MAX_SELF = 4  # keep at most this many past-self checkpoints in the pool


def write_pool(entries: list[dict], pool_path: str) -> None:
    Path(pool_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pool_path).write_text(json.dumps(entries, indent=2))


def train_gen(
    warm_ckpt: str,
    steps: int,
    out: str,
    seed: int,
    n_envs: int,
    log,
    pool_path: str,
    driver: str = "subproc",
    device: str = "auto",
) -> None:
    """Warm-start from ``warm_ckpt`` and best-respond to pfsp:POOL for ``steps``.

    ``driver`` selects the env backend: "subproc" (default, SubprocVecEnv with the
    opponent inline in each worker) or "batched" (single-process BatchedOpponentVecEnv
    that resolves all opponents in batched forwards — ~2-3x collection throughput,
    decision-preserving; see docs/worklog E36). The trained best-response is the
    same either way — only opponent-inference batching differs."""
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415

    if driver == "batched":
        from locma.envs.batched_selfplay import make_batched_opponent_vecenv  # noqa: PLC0415

        env = make_batched_opponent_vecenv(
            pool_path, n_envs, seed=seed, ldraft=LDRAFT, obs_variant="fx"
        )
    else:
        from locma.envs.training import _build_env  # noqa: PLC0415

        env = _build_env(
            f"pfsp:{pool_path}",
            seed,
            n_envs,
            both_seat=True,
            obs_mode="token-fx",
            draft_override=LDRAFT,
        )
    # load WITH the env (n_envs may differ from the saved model)
    model = MaskablePPO.load(resolve_path(warm_ckpt), env=env, device=device)
    log(f"  training best-response ({steps} steps, warm from {warm_ckpt})")
    model.learn(total_timesteps=steps, reset_num_timesteps=True)
    model.save(out)
    env.close()
    log(f"  saved {out}")


def eval_vs(net_spec: str, opp_specs: list[str], games: int, seed: int) -> dict[str, float]:
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    net = make_policy(net_spec)
    wr = {}
    for i, opp in enumerate(opp_specs):
        res = run_match(net, make_policy(opp), games=games, seed=seed + i * 1000)
        wr[opp] = round(res.win_rate_a, 3)
    return wr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--eval-games", type=int, default=100, help="pairs vs each pool member")
    ap.add_argument("--seed", type=int, default=14_000_000)
    ap.add_argument(
        "--start-gen", type=int, default=0, help="first generation index (naming + seed)"
    )
    ap.add_argument("--warm", default=E29, help="warm-start ckpt for the first gen of this run")
    ap.add_argument(
        "--driver",
        choices=["subproc", "batched"],
        default="subproc",
        help="env backend: subproc (inline opponent) or batched (single-process batched opponent)",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="SB3 learner device: auto|cpu|cuda|mps (tiny slim net often faster on cpu)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing chain: load its pool.json instead of reseeding SEED_POOL",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="isolate this chain's artifacts under runs/e36_<tag>_gen*.zip + runs/e36_<tag>/ "
        "so parallel seed chains don't clobber each other (empty = legacy runs/e36* paths)",
    )
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    run_dir = f"runs/e36{suffix}"
    pool_path = f"{run_dir}/pool.json"

    lines: list[str] = []

    def log(m: str) -> None:
        print(m, flush=True)
        lines.append(m)

    if args.resume and Path(pool_path).exists():
        pool = json.loads(Path(pool_path).read_text())
        log(f"resuming: loaded pool with {len(pool)} members from {pool_path}")
    else:
        pool = [dict(e) for e in SEED_POOL]
        write_pool(pool, pool_path)
    warm = args.warm  # first gen of this run warm-starts from here
    log(f"start-gen {args.start_gen}, warm from {warm}")
    history = []

    for g in range(args.start_gen, args.start_gen + args.generations):
        log(f"\n=== generation {g} ===")
        out = f"runs/e36{suffix}_gen{g}.zip"
        train_gen(
            warm,
            args.steps,
            out,
            args.seed + g,
            args.n_envs,
            log,
            pool_path,
            driver=args.driver,
            device=args.device,
        )
        new_spec = f"ppo:{out},{LDRAFT}"

        # eval the new net vs every pool member -> prioritised weights (PFSP)
        opp_specs = [e["spec"] for e in pool]
        wr = eval_vs(new_spec, opp_specs, args.eval_games, args.seed + 900 + g)
        log(f"  gen{g} win-rate vs pool: {json.dumps(wr)}")
        for e in pool:
            # prioritise opponents we're losing to: weight ~ (1 - winrate), floored
            e["weight"] = round(max(0.1, 1.0 - wr[e["spec"]]), 3)

        # admit the new net as a self member; cap the number of self checkpoints
        pool.append({"spec": new_spec, "weight": 1.0, "kind": "self"})
        selves = [e for e in pool if e.get("kind") == "self"]
        if len(selves) > MAX_SELF:
            drop = selves[0]["spec"]  # evict the oldest self
            pool = [e for e in pool if e["spec"] != drop]
            log(f"  evicted oldest self: {drop}")
        write_pool(pool, pool_path)
        history.append({"gen": g, "out": out, "wr_vs_pool": wr})
        warm = out  # next generation continues from this one

    hist_path = (
        f"{run_dir}/history.json"
        if args.start_gen == 0
        else f"{run_dir}/history_gen{args.start_gen}+.json"
    )
    Path(hist_path).write_text(json.dumps({"history": history, "log": lines}, indent=2))
    log(f"\nwrote {hist_path}")
    log(f"final net: {history[-1]['out'] if history else 'none'}")


if __name__ == "__main__":
    main()
