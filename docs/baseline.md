# Current baseline

The canonical pair-score matrix for the five built-in policies — row's win rate
vs column, `locma tournament random scripted greedy max-guard max-attack
--games 500 --seed 0 --matrix` (1000 games per pair, mirrored, `--seed 0`,
deterministic). This is the living reference; dated sections below are frozen
snapshots of past states.

|            | random | scripted | greedy | max-guard | max-attack |
|------------|--------|----------|--------|-----------|------------|
| random     | —      | 0.01     | 0.01   | 0.01      | 0.01       |
| scripted   | 0.99   | —        | 0.55   | 0.47      | 0.56       |
| greedy     | 0.99   | 0.45     | —      | 0.43      | 0.32       |
| max-guard  | 0.99   | 0.53     | 0.57   | —         | 0.57       |
| max-attack | 0.99   | 0.45     | 0.68   | 0.43      | —          |

Ranking by rating (openskill ordinal / Elo): `max-attack` (61.81 / 2846) >
`max-guard` (31.00 / 1976) > `greedy` (2.27 / 1253) > `scripted` (-15.75 / 886) >
`random` (-41.21 / 539). Note the pool is **non-transitive** — `max-guard` beats
`max-attack` and `scripted` beats `greedy` head-to-head, against the rating
order — so read the matrix, not just the ordinal.

---

# Baselines — 2026-06-24: new ground baselines + reworked `scripted`

_Date: 2026-06-24_

This version adds two ground-strategy baselines and reworks `scripted`. All
numbers reproducible with `--seed 0` (engine deterministic; the now-stochastic
policies are re-seeded per game, so logged games still replay byte-identically —
verified below). The original three-policy report is preserved unchanged below
the `---`.

## New baselines

- **`max-guard`** — draft prefers Guard creatures (then any creature, tie-broken
  by stat sum); "ground" battle: develop the board and swing at the enemy face,
  falling back to clearing Guards when the face is not a legal target.
- **`max-attack`** — draft prefers the highest-attack creature (creatures over
  items, tie-broken by defense); same ground battle.

## What changed

- **`scripted` reworked.** Was: fixed heuristic (always pick draft slot 0; first
  non-`Pass` legal action). Now: **random draft** + a fixed aggressive battle
  script — use green items on own creatures → attack the face (clear a Guard
  first when no face attack is legal) → summon creatures → use remaining items →
  pass, with targets chosen at random. Effect: `scripted` jumped from losing to
  `greedy` (old: greedy beat it 84%) to **beating `greedy` 55%** head-to-head,
  and from ~87% to **98.6%** vs `random`.
- The pre-existing three-policy numbers below are now **stale for `scripted`**
  (kept for historical continuity); use the tables in this section.

## Tournament ratings (5 policies)

`locma tournament random scripted greedy max-guard max-attack --games 500 --seed 0 --matrix`:

| Policy     | openskill (ordinal) | Elo  | p vs random |
|------------|---------------------|------|-------------|
| max-attack | 61.81               | 2846 | 4.48e-276   |
| max-guard  | 31.00               | 1976 | 2.57e-286   |
| greedy     | 2.27                | 1253 | 4.54e-282   |
| scripted   | -15.75              | 886  | 1.98e-270   |
| random     | -41.21              | 539  | —           |

Pair-score matrix (row's win rate vs column):

|            | random | scripted | greedy | max-guard | max-attack |
|------------|--------|----------|--------|-----------|------------|
| random     | —      | 0.01     | 0.01   | 0.01      | 0.01       |
| scripted   | 0.99   | —        | 0.55   | 0.47      | 0.56       |
| greedy     | 0.99   | 0.45     | —      | 0.43      | 0.32       |
| max-guard  | 0.99   | 0.53     | 0.57   | —         | 0.57       |
| max-attack | 0.99   | 0.45     | 0.68   | 0.43      | —          |

## Head-to-head win rates

`locma play A B --games 500 --seed 0` (1000 games each), win rate of A:

| Matchup                 | Win rate A | 95% CI        | binomial p |
|-------------------------|------------|---------------|------------|
| max-guard vs random     | 0.994      | 0.987–0.997   | 2.57e-286  |
| max-attack vs random    | 0.989      | 0.980–0.994   | 4.48e-276  |
| scripted vs random      | 0.986      | 0.977–0.992   | 1.98e-270  |
| max-attack vs greedy    | 0.677      | 0.647–0.705   | 1.59e-29   |
| max-guard vs greedy     | 0.569      | 0.538–0.599   | 1.43e-05   |
| max-guard vs max-attack | 0.569      | 0.538–0.599   | 1.43e-05   |
| scripted vs greedy      | 0.551      | 0.520–0.582   | 1.39e-03   |

## Non-transitivity

The pool is **not a clean total order** — the rating models (Elo and openskill
both assume transitive skill) hide a rock-paper-scissors structure:

- `max-guard` **beats** `max-attack` head-to-head (0.569), yet `max-attack` tops
  the table — it crushes `greedy` harder (0.68 vs `max-guard`'s 0.57), which the
  latent-skill models reward.
- `scripted` **beats** both `greedy` (0.55) and `max-attack` (0.56) head-to-head
  but **loses** to `max-guard` (0.47), so it rates *below* `greedy` despite
  beating it. Read the pair-score matrix, not just the ordinal, for this pool.

## Reproduce

```bash
uv run locma tournament random scripted greedy max-guard max-attack --games 500 --seed 0 --matrix
uv run locma play max-attack greedy    --games 500 --seed 0
uv run locma play max-guard  greedy    --games 500 --seed 0
uv run locma play max-attack max-guard --games 500 --seed 0
uv run locma play scripted   random    --games 500 --seed 0
uv run locma play greedy     scripted  --games 500 --seed 0

# Replay determinism still holds with the stochastic policies:
uv run locma play scripted random --games 50 --seed 0 --log v2.jsonl
uv run locma replay v2.jsonl --assert-hash   # exit 0
```

---

# Baseline Experiments

Reference baseline for the three built-in policies on single-lane LOCM 1.2.
All numbers below are reproducible: every run uses `--seed 0` and the engine is
deterministic (same seed + same policies → identical outcome and identical
trace). Generated 2026-06-23 with the `locma` CLI.

## Setup

- **Engine:** single-lane LOCM 1.2, 160-card set (`locma/data/cardlist.txt`).
- **Harness:** mirrored matches — for each seed `s` the pair is played twice,
  once with each policy in seat 0, so seat advantage cancels. A reported match
  of `--games N` is therefore `2N` games.
- **Policies:**
  - `random` — uniform random legal action (draft and battle).
  - `scripted` — fixed heuristic.
  - `greedy` — stat-based draft + greedy battle (attack/trade heuristic).
- **Statistics:** Wilson 95% CI, two-sided binomial test, SPRT (Wald LLR),
  Elo, and openskill (PlackettLuce, reported as the conservative ordinal
  `mu - 3*sigma`).

## Noise floor (luck baseline / resolution limit)

Each policy played against an independent copy of itself, `--games 1000`
(2000 games), `--seed 0`:

| Policy   | Win rate vs self | 95% CI        | Resolution limit (±) |
|----------|------------------|---------------|----------------------|
| random   | 0.500            | 0.478–0.522   | 0.022                |
| scripted | 0.500            | 0.478–0.522   | 0.022                |
| greedy   | 0.500            | 0.478–0.522   | 0.022                |

**Interpretation.** Because the harness mirrors every seed, a policy against an
identical copy of itself splits each seed's two games exactly — so self-play is
*exactly* 0.500 by construction, not by chance. What this measurement gives us
is the **sampling resolution at this sample size**: at 2000 games the 95% CI
half-width is ±0.022. Any claimed edge between two policies smaller than ~0.022
is indistinguishable from noise at this `n` — increase `--games` to resolve
finer differences.

## Head-to-head win rates

`locma play A B --games 500 --seed 0` (1000 games each), win rate of A:

| Matchup            | Win rate A | 95% CI        | binomial p   |
|--------------------|------------|---------------|--------------|
| greedy vs random   | 0.990      | 0.982–0.995   | 4.97e-278    |
| scripted vs random | 0.874      | 0.852–0.893   | 2.46e-138    |
| greedy vs scripted | 0.836      | 0.812–0.858   | 5.00e-109    |

All three gaps are far larger than the ±0.022 noise floor and the binomial
p-values are effectively zero: the ordering **greedy > scripted > random** is
unambiguous.

## Tournament ratings

`locma tournament random scripted greedy --games 500 --seed 0 --matrix`:

| Policy   | openskill (ordinal) | Elo  | p vs random  |
|----------|---------------------|------|--------------|
| greedy   | 49.83               | 2376 | 4.97e-278    |
| scripted | 12.40               | 1385 | 2.46e-138    |
| random   | -20.94              | 739  | —            |

Pair-score matrix (row's win rate vs column):

|          | random | scripted | greedy |
|----------|--------|----------|--------|
| random   | —      | 0.13     | 0.01   |
| scripted | 0.87   | —        | 0.16   |
| greedy   | 0.99   | 0.84     | —      |

Both rating systems agree on the ordering. openskill is the primary metric (it
models uncertainty); Elo is retained for continuity.

## SPRT (sequential testing)

`locma sprt A --vs B` with H0: winrate = 0.5, H1: winrate = 0.6,
α = β = 0.05, batch 20, `--seed 0`:

| Matchup            | Verdict    | Win rate | n   |
|--------------------|------------|----------|-----|
| greedy vs random   | accept_h1  | 1.000    | 40  |
| scripted vs random | accept_h1  | 0.875    | 40  |
| greedy vs scripted | accept_h1  | 0.875    | 40  |

Every matchup is decided in the **first batch** (n = 40): the effects are so
large the sequential test crosses the H1 boundary immediately. SPRT's value
shows on near-threshold comparisons (e.g. a candidate that is only slightly
better than its baseline), where it stops far earlier than a fixed-n test —
these reference matchups simply have no ambiguity to resolve.

## Replay determinism

`locma play greedy scripted --games 50 --seed 0 --log <file>` then
`locma replay <file> --assert-hash` → **exit 0** across all 100 logged games:
every game's recomputed content hash matches the stored hash. The trace +
hash pipeline is byte-stable, so logged games are a reliable regression anchor
for future engine or policy changes.

## Reproduce

```bash
# Noise floor (per policy)
uv run locma noise-floor random   --games 1000 --seed 0
uv run locma noise-floor scripted --games 1000 --seed 0
uv run locma noise-floor greedy   --games 1000 --seed 0

# Head-to-head
uv run locma play greedy   random   --games 500 --seed 0
uv run locma play scripted random   --games 500 --seed 0
uv run locma play greedy   scripted --games 500 --seed 0

# Tournament + matrix
uv run locma tournament random scripted greedy --games 500 --seed 0 --matrix

# SPRT
uv run locma sprt greedy   --vs random   --max-games 2000 --seed 0
uv run locma sprt scripted --vs random   --max-games 2000 --seed 0
uv run locma sprt greedy   --vs scripted --max-games 2000 --seed 0

# Replay determinism
uv run locma play greedy scripted --games 50 --seed 0 --log base.jsonl
uv run locma replay base.jsonl --assert-hash
```

## Takeaways

- **Ordering: greedy > scripted > random**, confirmed by win rate, both rating
  systems, and SPRT — all with p-values at the floating-point floor.
- **greedy beats random 99%** and **scripted 84%**; **scripted beats random
  87%**. greedy is the strongest reference opponent for evaluating new policies.
- **Resolution at 2000 games is ±0.022.** Any new policy whose edge over a
  baseline is smaller than this needs more games (or SPRT) to call.
- **Replay is byte-stable**, so these baselines can be regression-checked after
  any engine/policy change.
