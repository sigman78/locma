"""FPS bench: live drafting vs a cached deck pool, and the amortized overhead.

Measures game throughput two ways, identical battle policies both times:
  * LIVE   — full ``run_game``: ldraft drafts both seats (~60 policy calls) then
             the battle plays out.
  * CACHED — ``run_battle_from_decks`` on decks sampled from a ``DeckPool``: the
             draft is skipped entirely.

The speedup is the draft's share of per-game cost. Then it reports the pool's
amortized generation overhead so you can see it stays under the budget the guard
enforces (generation can never creep toward the live drafting cost).

    python scripts/e36_deckpool_bench.py --games 60 --pool-size 200
"""

from __future__ import annotations

import argparse
import random
import time

from locma.core.engine import run_battle_from_decks, run_game
from locma.data.cards_db import load_cards
from locma.envs.deckpool import DeckPool
from locma.policies.registry import make_policy

NET = "ppo:depot:e36/e36_gen7.zip,depot:ldraft/ldraft_s0.zip"
OPP = "ppo:depot:e29slim/e29slim_s0.zip,depot:ldraft/ldraft_s0.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=60, help="games per mode (serial)")
    ap.add_argument("--pool-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=80_000_000)
    args = ap.parse_args()

    cards = load_cards()
    net = make_policy(NET)
    opp = make_policy(OPP)

    # --- LIVE: full run_game (ldraft drafts both seats) ---
    t0 = time.perf_counter()
    for i in range(args.games):
        run_game(net, opp, seed=args.seed + i, cards=cards)
    live_s = time.perf_counter() - t0
    live_fps = args.games / live_s

    # --- Build the cached pool (time it — this is the amortized cost) ---
    t0 = time.perf_counter()
    pool = DeckPool.generate(size=args.pool_size, seed=args.seed, cards=cards)
    gen_s = time.perf_counter() - t0
    # one-deck cost, for the amortization arithmetic
    per_deck_s = gen_s / args.pool_size

    # --- CACHED: sample decks + run_battle_from_decks ---
    rng = random.Random(args.seed)
    t0 = time.perf_counter()
    for i in range(args.games):
        d0, d1 = pool.sample_pair(rng)
        run_battle_from_decks(d0, d1, net, opp, seed=args.seed + i, cards=cards)
    cached_s = time.perf_counter() - t0
    cached_fps = args.games / cached_s

    speedup = cached_fps / live_fps
    draft_share = 1 - cached_s / live_s  # fraction of live per-game cost that was draft

    net_name = NET.split(",", maxsplit=1)[0]
    opp_name = OPP.split(",", maxsplit=1)[0]
    print("\n================ DeckPool FPS bench ================")
    print(f"battle: {net_name}  vs  {opp_name}   ({args.games} games/mode, serial)")
    print(f"  LIVE   (draft+battle): {live_fps:6.2f} games/s  ({live_s:.1f}s)")
    print(f"  CACHED (battle only) : {cached_fps:6.2f} games/s  ({cached_s:.1f}s)")
    print(f"  speedup: {speedup:.2f}x   (draft was {draft_share:.0%} of per-game cost)")

    print("\n---- amortized generation overhead ----")
    print(
        f"  pool build: {args.pool_size} decks in {gen_s:.1f}s  ({per_deck_s * 1000:.1f} ms/deck)"
    )
    # break-even: after how many games does the pool build pay for itself?
    saved_per_game = live_s / args.games - cached_s / args.games
    breakeven = gen_s / saved_per_game if saved_per_game > 0 else float("inf")
    print(f"  break-even: pool pays for itself after ~{breakeven:.0f} games")
    # budget headroom at the default eps=0.02: matches needed before a refresh is allowed
    eps = pool.gen_budget_frac
    m_floor = int(args.pool_size / (2 * eps))
    print(
        f"  budget: at eps={eps}, the guard allows a refresh only after "
        f"~{m_floor:,} matches (initial pool amortized) — so cumulative gen "
        f"stays <= {eps:.0%} of live drafting by construction"
    )
    # sanity: mixture produced varied decks
    uniq = len({tuple(sorted(d)) for d in pool.decks})
    print(f"  pool diversity: {uniq}/{args.pool_size} unique decks (80/20 ldraft/random mix)")


if __name__ == "__main__":
    main()
