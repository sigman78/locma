# Benchmarking draft (deck-building) strategy

How to measure a **draft policy** in isolation — separated from the battle policy
that pilots the deck — and what the measurement says is the best drafting strategy
under the current rules. The tool is `locma draft-bench`
(`locma/harness/draft_bench.py`); the statistics reuse `docs/experiments.md`.

## The problem: drafting is entangled with battling

A `Policy` is two halves — a draft half (which card to pick) and a battle half
(how to play). The win rate of a full policy reflects *both*. The older "PPO ×
draft sweep" (`docs/baseline.md`, 2026-06-25) measured a draft by pairing it with a
fixed battle net and playing it against the built-in **baselines** — but each
baseline brings its *own* battle AND draft, so that number conflates "is this a
better deck?" with "does this battle net beat that opponent's battle script?".
To rank drafts you want only the draft to vary.

## The key insight: the draft is a paired comparison

The draft phase (`locma/core/draft.py`) deals **both seats from the same shared
triplet stream**, one triplet per round, and a pick does **not** deplete the
offer. So on a fixed seed both seats are offered *identical cards* in the same
order. Therefore:

> If both seats are piloted by the **same battle policy** and differ **only** in
> their draft policy, the battle skill cancels and the head-to-head win rate is
> attributable purely to *which deck was drafted from identical offers*.

This is a clean **paired comparison** (same card stream, same pilot) — far lower
variance and far less confounded than playing drafted decks against the
heterogeneous baselines.

### Calibration guarantee: a self-duel is exactly 0.500

A draft played against *itself* (same policy both seats, same pilot) wins
**exactly 50%** — not approximately. For each seed the two behaviourally-identical
policies produce the same winner-seat in both mirrored games, so the seat-0 player
wins exactly one of the mirrored pair. The mirror cancels seat advantage
*perfectly*. This is an exact, build-time calibration check: any deviation from
0.500 in a self-duel is a benchmark bug. (Verified for `ground`, `greedy`, the
stochastic state-cloning `azlite`, and the net-backed `netdmcts` pilots — all
exactly 0.5000; the last confirms even a neural oracle plays deterministically.)

## The method

`duel(draft_a, draft_b, battle, games, seed)` builds
`Composer(battle, draft_a)` vs `Composer(battle, draft_b)`, plays a mirrored match
(`run_match` already swaps seats), and reports `draft_a`'s Wilson-CI'd win rate.
`round_robin` runs all pairs once (the reverse cell is the exact complement, so
half the games suffice) and ranks by **average win rate vs the field** — a
Copeland-style score that, unlike Elo/openskill, is not fooled by a lopsided win
over one weak draft (and `docs/baseline.md` documents at length how the latent-skill
ratings mislead on this kit — read the matrix).

```bash
# rank all built-in drafts under one pilot, with the matrix
uv run locma draft-bench --battle ground --games 300
uv run locma draft-bench balanced max-guard max-attack --battle ppo:runs/ppo-shuffled-pool.zip --games 150
```

## Choosing the pilot — the load-bearing decision

The deck that wins **depends on who plays it**. The benchmark makes this concrete,
and it is the single most important methodological point:

- **Weak heuristic pilots disagree with each other**, because each imposes its own
  style. Under the `ground` pilot (pure face aggression) the curve+Guard decks win;
  under the `greedy` battle pilot (trades into creatures, summons the biggest drop)
  a high-attack deck wins and Guard-heavy decks *lose*. The ranking literally
  inverts. A single weak pilot is therefore an **unreliable** draft benchmark.
- **Use a strong pilot** so the ranking reflects deck strength under *skilled*
  play, not a heuristic's quirks. Options, in order of trust:
  - `netdmcts:8,40,1.5,<net>` — the **strongest policy in the kit** *and* fair (PUCT
    over determinized worlds with a trained net oracle — no hidden info). The most
    trustworthy verdict: under it `balanced` sweeps every head-to-head and replicates
    across seeds. Slowest (~13 s/game); parallelize across single-threaded processes
    (`torch.set_num_threads(1)`; the small net gets no benefit from intra-op threads).
  - `dmcts` — fair but with a *heuristic* rollout oracle, so weaker than `netdmcts`;
    it over-credits item-heavy decks (`random` near-ties `balanced`).
  - `azlite:100` — strong and fast (~0.5 s/game). It *cheats* (perfect foresight),
    but **both seats cheat identically**, so the cheat cancels in the mirror and
    what remains is "which deck wins under strong play". Self-duel is still exactly
    0.5, confirming the symmetry. Like `dmcts` it tilts toward item decks.
  - `ppo:<model>` — the learned battle net: the **deployment** pilot ("which deck
    is best for the agent we actually ship?").
- **Report under several pilots.** A draft that is best across `ground`, the
  deployment `ppo`, and a strong fair searcher is robustly best; one that only wins
  under a single aggressive pilot is overfit to that pilot's style.

## Results

Full pilot-by-pilot matrices are in `docs/baseline.md` ("Draft-bench — 2026-06-28").
Ranking by average win rate vs field (higher = better deck), all 7 built-in drafts:

| draft       | ground | ppo (deploy) | dmcts (fair) | netdmcts (fair, top) | azlite (foresight) | greedy-battle |
|-------------|:------:|:------------:|:------------:|:--------------------:|:------------------:|:-------------:|
| **balanced**| **0.650** | **0.624** | **0.647** | **0.610** | 0.613 | 0.567 |
| random      | 0.471 | 0.476 | 0.643 | 0.543 | **0.674** | 0.080 |
| max-guard   | 0.591 | 0.535 | 0.450 | 0.453 | 0.526 | 0.393 |
| max-attack  | 0.527 | 0.542 | 0.460 | 0.463 | 0.457 | **0.694** |
| max-defense | 0.434 | 0.466 | 0.453 | 0.480 | 0.413 | 0.576 |
| weighted    | 0.414 | 0.407 | 0.420 | 0.467 | 0.428 | 0.574 |
| greedy      | 0.412 | 0.450 | 0.427 | 0.483 | 0.390 | 0.616 |

Headline:
- **`balanced` is the robustly best draft** — the Condorcet winner (beats every
  other draft head-to-head) under `ground`, `ppo`, **and `netdmcts`** (the kit's
  strongest fair policy, where it sweeps all six head-to-heads 0.56–0.64 and
  **replicates across two seeds**). Under the two *weaker* search pilots it is #1 by
  avg win rate (`dmcts`, edging `random` 0.647 vs 0.643 — a tie) and a close 2nd
  under `azlite`, but not the Condorcet winner there (it loses the `random`
  head-to-head; see below).
- The shipped reference `greedy` draft and `weighted` rank **at or below a random
  draft** under every serious pilot — confirming the long-standing finding that
  `greedy` is the worst draft and the deck is the lever.
- **Pilot choice changes the ranking — use a strong one.** The weak `greedy`-battle
  pilot inverts it (max-attack #1); `random` looks competitive only under the
  *weaker/cheating* searchers (it carries items they use well) and is mediocre under
  `ground`/`ppo`. It edges `balanced` head-to-head at seed 0 under those weak
  searchers (`balanced` wins only 0.40 azlite / 0.44 dmcts) but that does **not**
  replicate (fresh seed 0.49 / 0.59). Under the **strongest fair pilot (`netdmcts`)**
  the effect vanishes: `random` is a clear #2 and *loses* to `balanced` 0.39 at both
  seeds — so "random rides its items" is a weak/cheating-searcher artifact, not real
  deck strength.
- **No tuning lever robustly beats `balanced`.** A cheaper curve wins only under
  `ground` (ties under `ppo` — overfit). A direct item-content probe (lowering
  `item_discount` to add premium removal) **ties** `balanced` under both `azlite` and
  `dmcts` and is no better under `ppo` — so the item-light recipe is robust, not an
  overfit. `balanced` is the best drafting strategy under the current rules: the
  Condorcet winner under `ground`, `ppo`, and the kit's strongest fair searcher
  (`netdmcts`, replicated across seeds), #1 or tied-#1 under every credible pilot,
  and un-improvable by the levers tested.
