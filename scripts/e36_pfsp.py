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

Pool entries may carry ``"pin_share": s`` — such members skip PFSP reweighting and
their weight is re-solved each generation so their sampling probability stays
exactly ``s`` (dose-controlled anchors, E37c dose-response arms).

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
import shutil
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


# obs_mode ("token-fx"/"token-fxh") -> the batched driver's obs_variant string.
_OBS_VARIANT = {"token-fx": "fx", "token-fxh": "fxh"}


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
    deck_pool_path: str | None = None,
    obs_mode: str = "token-fx",
    warm_widen: bool = False,
) -> None:
    """Warm-start from ``warm_ckpt`` and best-respond to pfsp:POOL for ``steps``.

    ``driver`` selects the env backend: "subproc" (default, SubprocVecEnv with the
    opponent inline in each worker) or "batched" (single-process BatchedOpponentVecEnv
    that resolves all opponents in batched forwards — ~2-3x collection throughput,
    decision-preserving; see docs/worklog E36). The trained best-response is the
    same either way — only opponent-inference batching differs.

    ``obs_mode`` (E42) selects the token-obs variant: "token-fx" (default, the
    pre-E42 byte-identical path) or "token-fxh" (fx tokens + the public-history
    ``hist`` branch). The batched driver maps this to ``obs_variant`` "fx"/"fxh"
    only if ``make_batched_opponent_vecenv`` actually accepts it; combining
    "token-fxh" with ``--driver batched`` raises a clear error otherwise.

    ``warm_widen`` (E42) builds the model via ``locma.envs.warmstart.warm_start``
    instead of ``MaskablePPO.load(..., env=env)``: a FRESH model at ``warm_ckpt``'s
    hyperparameters, its weights copied (the new ``hist`` branch, if any,
    zero-init'd) and a fresh optimizer — needed the first time a chain crosses
    obs spaces (e.g. fx -> fxh). Use it for the first generation of a run only;
    later generations warm from the arm's own previous-gen checkpoint (already in
    the target obs space) via the normal ``MaskablePPO.load``."""
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.depot import resolve_path  # noqa: PLC0415

    if driver == "batched":
        if deck_pool_path is not None:
            raise ValueError("--deck-pool requires the default 'subproc' driver (not 'batched')")
        from locma.envs.batched_selfplay import make_batched_opponent_vecenv  # noqa: PLC0415

        obs_variant = _OBS_VARIANT[obs_mode]
        try:
            env = make_batched_opponent_vecenv(
                pool_path, n_envs, seed=seed, ldraft=LDRAFT, obs_variant=obs_variant
            )
        except Exception as exc:  # noqa: BLE001
            if obs_variant == "fxh":
                raise ValueError(
                    "--driver batched does not support --obs-mode token-fxh yet "
                    f"(make_batched_opponent_vecenv rejected obs_variant='fxh': {exc}); "
                    "use --driver subproc for token-fxh"
                ) from exc
            raise
    else:
        from locma.envs.training import _build_env  # noqa: PLC0415

        # deck_pool_path set -> envs sample cached decks (no live ldraft draft);
        # draft_override is then a no-op but kept harmless.
        env = _build_env(
            f"pfsp:{pool_path}",
            seed,
            n_envs,
            both_seat=True,
            obs_mode=obs_mode,
            draft_override=LDRAFT,
            deck_pool_path=deck_pool_path,
        )
    warm_path = resolve_path(warm_ckpt)
    if warm_widen:
        from locma.envs.warmstart import warm_start  # noqa: PLC0415

        log(
            f"  warm-widen: building a fresh model from {warm_ckpt} "
            f"(obs_mode={obs_mode}, weights copied, fresh optimizer)"
        )
        model = warm_start(warm_path, env, obs_mode=obs_mode, device=device)
    else:
        log(f"  loading {warm_ckpt} with env (obs_mode={obs_mode})")
        model = MaskablePPO.load(warm_path, env=env, device=device)
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
    ap.add_argument(
        "--deck-pool",
        default=None,
        help="path to a cached deck-pool JSON; if set, envs sample pre-drafted decks "
        "instead of live ldraft drafting (removes ~60 draft calls/game). Auto-generated "
        "with the 80/20 ldraft/random mixture if the file is absent. Subproc driver only.",
    )
    ap.add_argument(
        "--deck-pool-size", type=int, default=2000, help="decks to pre-draft if generating"
    )
    ap.add_argument(
        "--obs-mode",
        choices=["token-fx", "token-fxh"],
        default="token-fx",
        help="training obs encoding (E42): 'token-fx' (default, byte-identical to "
        "pre-E42 behavior) or 'token-fxh' (fx tokens + the public-history hist branch)",
    )
    ap.add_argument(
        "--warm-widen",
        action="store_true",
        help="(E42) build the FIRST generation's model via "
        "locma.envs.warmstart.warm_start (fresh model at the warm ckpt's "
        "hyperparameters, weights copied, hist branch zero-init, fresh optimizer) "
        "instead of MaskablePPO.load(..., env=env); later generations of this run "
        "warm from the arm's own previous gen via the normal load",
    )
    ap.add_argument(
        "--pool-from",
        default=None,
        help="(E42) if set and this --tag's pool.json does not exist yet, copy this "
        "pool file into the run dir before starting, e.g. "
        "'--tag e42h --pool-from runs/e36/pool.json --resume' continues the x86 "
        "terminal pool under a new tag. Does not change --resume semantics otherwise.",
    )
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    run_dir = f"runs/e36{suffix}"
    pool_path = f"{run_dir}/pool.json"

    lines: list[str] = []

    def log(m: str) -> None:
        print(m, flush=True)
        lines.append(m)

    if args.pool_from and not Path(pool_path).exists():
        Path(pool_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.pool_from, pool_path)
        log(f"pool-from: copied {args.pool_from} -> {pool_path}")

    if args.resume and Path(pool_path).exists():
        pool = json.loads(Path(pool_path).read_text())
        log(f"resuming: loaded pool with {len(pool)} members from {pool_path}")
    else:
        pool = [dict(e) for e in SEED_POOL]
        write_pool(pool, pool_path)
    # Cached deck pool (optional): load or generate once; workers load it read-only.
    deck_pool = None
    deck_pool_path = args.deck_pool
    if deck_pool_path is not None:
        from locma.envs.deckpool import DeckPool  # noqa: PLC0415

        if Path(deck_pool_path).exists():
            deck_pool = DeckPool.load(deck_pool_path)
            log(f"deck-pool: loaded {len(deck_pool.decks)} decks from {deck_pool_path}")
        else:
            log(f"deck-pool: generating {args.deck_pool_size} decks (80/20 ldraft/random)...")
            deck_pool = DeckPool.generate(args.deck_pool_size, seed=args.seed)
            deck_pool.save(deck_pool_path)
            log(f"deck-pool: wrote {deck_pool_path}")

    warm = args.warm  # first gen of this run warm-starts from here
    log(f"start-gen {args.start_gen}, warm from {warm}")
    history = []

    for g in range(args.start_gen, args.start_gen + args.generations):
        log(f"\n=== generation {g} ===")
        out = f"runs/e36{suffix}_gen{g}.zip"
        gen_warm_widen = args.warm_widen and g == args.start_gen  # first gen of THIS run only
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
            deck_pool_path=deck_pool_path,
            obs_mode=args.obs_mode,
            warm_widen=gen_warm_widen,
        )
        new_spec = f"ppo:{out},{LDRAFT}"

        # Deck-pool refresh (budget-guarded): after each generation, tell the pool
        # how many matches it served (~steps / agent-decisions-per-game) and let it
        # roll a fresh slice IFF cumulative generation stays under the cost budget.
        # Workers reload the refreshed file at the next generation.
        if deck_pool is not None:
            deck_pool.record_matches(args.steps // 30)  # ~30 agent decisions/game
            replaced = deck_pool.maybe_refresh(seed=args.seed + 7000 + g)
            deck_pool.save(deck_pool_path)
            st = deck_pool.stats()
            gate = "deferred" if replaced == 0 else "ok"
            log(
                f"  deck-pool: refreshed {replaced} decks "
                f"(amortized gen {st['amortized_frac']} vs {st['budget_frac']} budget, {gate})"
            )

        # eval the new net vs every pool member -> prioritised weights (PFSP)
        opp_specs = [e["spec"] for e in pool]
        wr = eval_vs(new_spec, opp_specs, args.eval_games, args.seed + 900 + g)
        log(f"  gen{g} win-rate vs pool: {json.dumps(wr)}")
        for e in pool:
            if "pin_share" in e:
                continue  # dose-pinned member (E37c): its share is the treatment variable
            # prioritise opponents we're losing to: weight ~ (1 - winrate), floored
            e["weight"] = round(max(0.1, 1.0 - wr[e["spec"]]), 3)

        # admit the new net as a self member; cap the number of self checkpoints
        pool.append({"spec": new_spec, "weight": 1.0, "kind": "self"})
        selves = [e for e in pool if e.get("kind") == "self"]
        if len(selves) > MAX_SELF:
            drop = selves[0]["spec"]  # evict the oldest self
            pool = [e for e in pool if e["spec"] != drop]
            log(f"  evicted oldest self: {drop}")
        # re-solve pinned weights against the post-admission pool so each pinned
        # member's sampling probability equals its pin_share exactly next gen
        pinned = [e for e in pool if "pin_share" in e]
        if pinned:
            rest = sum(e["weight"] for e in pool if "pin_share" not in e)
            s_tot = sum(float(e["pin_share"]) for e in pinned)
            for e in pinned:
                e["weight"] = round(float(e["pin_share"]) / (1.0 - s_tot) * rest, 3)
                log(f"  pinned {e['spec']} at share {e['pin_share']} (weight {e['weight']})")
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
