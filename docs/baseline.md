# Current baseline

The canonical pair-score matrix for the five built-in baseline policies — row's
win rate vs column, `locma tournament random scripted greedy max-guard
max-attack --games 500 --seed 0 --matrix` (1000 games per pair, mirrored,
`--seed 0`). This is the living reference; dated sections below are frozen
snapshots. Refreshed 2026-06-26 after the new shuffled `DraftSource` default
(PR #31): the draft pool is now a shuffle of the whole 160-card space duplicated
`copies=2` (each card offered at most twice), replacing the old
uniform-with-replacement sampling. Cells shifted by 1–3 points — the same order
of magnitude as the ADR-0003 RNG split — and the **ordering is unchanged**. The
prior uniform-pool matrix is frozen in the 2026-06-26 snapshot below. The search
(`mcts`) and learned (`ppo`) policies live in the dated sections further down.

|            | random | scripted | greedy | max-guard | max-attack |
|------------|--------|----------|--------|-----------|------------|
| random     | —      | 0.01     | 0.01   | 0.01      | 0.02       |
| scripted   | 0.99   | —        | 0.55   | 0.51      | 0.61       |
| greedy     | 0.99   | 0.45     | —      | 0.45      | 0.32       |
| max-guard  | 0.99   | 0.49     | 0.55   | —         | 0.55       |
| max-attack | 0.98   | 0.39     | 0.68   | 0.45      | —          |

Ranking by rating (openskill ordinal / Elo): `max-attack` (60.83 / 2845) >
`max-guard` (29.69 / 1967) > `greedy` (1.04 / 1247) > `scripted` (-16.13 / 902) >
`random` (-42.27 / 540). Note the pool is **non-transitive** — and more so under
the shuffled pool: `scripted` (rated 4th) beats `greedy` (0.55), `max-guard`
(0.51), **and** `max-attack` (0.61) head-to-head, yet rates below all three; and
`max-guard` beats `max-attack` (0.55) against the rating order. Read the matrix,
not just the ordinal.

---

# Baselines — 2026-06-26: full-roster tournament (baselines + search + PPO) — azlite tops; rating estimator fixed

_Date: 2026-06-26_

The whole bunch in one bracket: the five built-in baselines plus all three search
policies (`mcts:100` cheating perfect-info, `azlite:100` AlphaZero-lite — **also
cheating** (perfect-foresight), see `docs/searchers-fiasco.md` — and `dmcts` the
one fair determinized MCTS) and the strongest learned policy
(`ppo:runs/ppo-shuffled-pool.zip`). One round-robin so openskill/Elo rate everyone
against everyone — `locma tournament … --games 200 --seed 0 --reference random
--matrix`, 400 games per pair (mirrored), 36 pairs, 14,400 games. This is the
**first appearance of `dmcts` numbers** anywhere in this file.

## Ratings (openskill ordinal / Elo)

Computed with the **order-free** rating estimator (Bradley-Terry Elo +
shuffle-averaged openskill) introduced in this change — see "why they didn't
agree before" below.

| policy                          | openskill | elo  | p vs random |
|---------------------------------|-----------|------|-------------|
| `azlite:100`                    | 29.05     | 1755 | 7.7e-121    |
| `mcts:100`                      | 28.72     | 1748 | 7.7e-121    |
| `dmcts`                         | 28.28     | 1732 | 7.7e-121    |
| `ppo` (shuffled-pool)           | 22.75     | 1578 | 3.1e-118    |
| `scripted`                      | 21.76     | 1552 | 6.2e-116    |
| `max-guard`                     | 20.88     | 1529 | 8.3e-114    |
| `max-attack`                    | 20.09     | 1507 | 8.3e-114    |
| `greedy`                        | 16.92     | 1438 | 8.2e-112    |
| `random`                        | -20.67    | 661  | —           |

## Pair-score matrix (row's win rate vs column)

`ppo` = `ppo:runs/ppo-shuffled-pool.zip`.

|            | random | scripted | greedy | max-guard | max-attack | mcts:100 | azlite:100 | dmcts | ppo  |
|------------|--------|----------|--------|-----------|------------|----------|------------|-------|------|
| random     | —      | 0.01     | 0.01   | 0.01      | 0.01       | 0.00     | 0.00       | 0.00  | 0.00 |
| scripted   | 0.99   | —        | 0.57   | 0.47      | 0.59       | 0.30     | 0.31       | 0.32  | 0.41 |
| greedy     | 0.99   | 0.43     | —      | 0.43      | 0.33       | 0.07     | 0.18       | 0.07  | 0.35 |
| max-guard  | 0.99   | 0.53     | 0.56   | —         | 0.58       | 0.20     | 0.23       | 0.20  | 0.42 |
| max-attack | 0.99   | 0.41     | 0.67   | 0.42      | —          | 0.20     | 0.20       | 0.23  | 0.38 |
| mcts:100   | 1.00   | 0.70     | 0.93   | 0.80      | 0.81       | —        | 0.45       | 0.54  | 0.71 |
| azlite:100 | 1.00   | 0.69     | 0.82   | 0.77      | 0.81       | 0.56     | —          | 0.60  | 0.76 |
| dmcts      | 1.00   | 0.68     | 0.93   | 0.80      | 0.77       | 0.46     | 0.40       | —     | 0.74 |
| ppo        | 1.00   | 0.59     | 0.65   | 0.57      | 0.62       | 0.28     | 0.24       | 0.26  | —    |

## Ratings now agree with the matrix — and why they didn't before

Both the openskill ordinal and Elo now rank the roster in exactly its
head-to-head order: **`azlite` > `mcts` > `dmcts` > `ppo` > {baselines} >
`random`**. `azlite:100` — the **only policy that wins every head-to-head** (beats
cheating `mcts` 0.56, `dmcts` 0.60, `ppo` 0.76, and all five baselines
0.69–1.00) — tops both. `ppo` beats all five baselines (0.57–0.65) but loses every
game to the three searches (0.24–0.28), so it lands 4th, just above the baselines.

**This was not true under the old estimator, and the difference was a bug, not
non-transitivity.** The original `elo_from_results` / `openskill_from_results`
updated ratings in a single sequential sweep, so the result depended on the
*order* games were fed — and the tournament feeds them grouped by pair (all of
A's wins, then B's). On this exact matrix the old single-pass estimator rated
**`ppo` #1 (openskill 64.87 / Elo 3177) and `azlite` #3** — almost the reverse of
the truth. Two checks pinned the cause: (1) merely **shuffling the game order**
moved `azlite` 3rd → 1st and `ppo` 1st → 4th, and (2) a convergent
**Bradley-Terry** fit recovered the matrix order exactly. This change replaces
both estimators with order-free fits — **Elo via Bradley-Terry MM**, **openskill
via seeded shuffle-averaging** (`locma/stats/`) — so a rating is now a function of
the results, not their order.

The genuine non-transitivity that remains is small and confined to the baselines:
`scripted` beats `greedy` (0.57) and `max-attack` (0.59) but loses to `max-guard`
(0.47), so no scalar can order those three perfectly — hence their clustered
20–22 ordinals. **Read the matrix for the baseline rock-paper-scissors; the gross
ordering the ratings now report is correct.** (The "ratings rank PPO #1 — read the
matrix" refrain in the dated sections below is partly this same single-pass
artifact; see the methodology note.)

## Strength order, and what `dmcts` adds

Head-to-head ranking, now matched by the ratings above: **`azlite` > `mcts` >
`dmcts` > `ppo` > {ground baselines} > `random`**. The strong cluster is cleanly
ordered; the ground baselines keep their familiar rock-paper-scissors (`scripted`
beats `greedy` 0.57 and `max-attack` 0.59 but loses to `max-guard` 0.47).

The three searches all crush the ground baselines far harder than `ppo` does
(avg-hard3 over `scripted`/`max-guard`/`max-attack`: `mcts` **0.770**, `azlite`
**0.757**, `dmcts` **0.750**, vs `ppo` **0.593**). Note `mcts` edges `azlite` on
the baseline sweep (0.770 vs 0.757) yet *loses* to it head-to-head (0.45) — even
avg-hard3 doesn't capture that `azlite` is undefeated; only the matrix does.

`dmcts` (determinized MCTS — sample `K` opponent hands, run MCTS per world) debuts
and is the **only fair searcher in the roster**: it beats every baseline and `ppo`
(0.74) but sits below both `mcts` and `azlite` head-to-head. That gap is not pure
search quality — **`mcts` and `azlite` cheat**: both clone the real `GameState`, so
their search sees the opponent's hidden hand *and* both decks' future draw order
(perfect foresight), while `dmcts` only samples what it cannot see. So `azlite`
dominating `dmcts` (0.60) is partly an information advantage, not skill (see
`docs/searchers-fiasco.md`). The one honest search-vs-reactive number here is
**`dmcts` vs `ppo` = 0.74**.

Cross-check: `azlite` vs `ppo` (0.76) and vs `mcts` (0.56) reproduce the prior
azlite section's 0.760 / 0.570 (200-game) within noise; its avg-hard3 0.757 here
matches its solo 0.741.

## Methodology note — rating tables in older sections predate this fix

The rating (openskill / Elo) tables in the **dated sections below were produced by
the order-dependent single-pass estimator** and are therefore unreliable — not
just in magnitude but sometimes in *ordering* (the PPO sections' "ratings rank PPO
#1 while it loses to 3/5 baselines" is largely the same artifact, not pure
non-transitivity). The **pair-score matrices in those sections are unaffected** —
they are direct win rates. Only this section's ratings use the order-free
estimator; re-rating an old matrix with it will move the numbers (and may reorder
them). Read older ordinals with that caveat; trust the matrices.

## Reproduce

```bash
uv run locma tournament random scripted greedy max-guard max-attack \
  mcts:100 azlite:100 dmcts ppo:runs/ppo-shuffled-pool.zip \
  --games 200 --seed 0 --reference random --matrix
```

(`ppo:` needs the `[ml]` extra and a local `runs/ppo-shuffled-pool.zip`; the
search policies are self-contained. `dmcts` at ~0.8 s/game is the bottleneck —
the full bracket is ~80 min.)

---

# Baselines — 2026-06-26: PPO curriculum endpoint vs `azlite` — no movement

_Date: 2026-06-26_

Follow-up to the full-roster result and `docs/ppo-review.md` §8: does adding a
strong searcher as a **final curriculum phase** lift the PPO? We warm-started the
strongest zoo model (`runs/ppo-shuffled-pool.zip`, the 800k both-seat zoo) and
continued it for **+200k steps against `azlite:100`** as the opponent — the new
search-as-training-opponent path (the `BattleEnv` now passes the forward-model
state; see `docs/searchers-fiasco.md`). Evaluated paired with the `balanced` draft
vs the pool, 300 games/cell, `--seed 0`:

| model                          | random | greedy | scripted | max-guard | max-attack | azlite | **avg-hard3** |
|--------------------------------|--------|--------|----------|-----------|------------|--------|---------------|
| base (shuffled-pool zoo)       | 0.997  | 0.650  | 0.583    | 0.580     | 0.620      | 0.237  | **0.594**     |
| **endpoint (+200k vs azlite)** | 0.990  | 0.687  | 0.530    | 0.613     | 0.647      | 0.270  | **0.597**     |

**No movement.** avg-hard3 0.594 → 0.597 (+0.003) is flat; the per-cell shuffle
(`scripted` −0.05, `max-guard`/`max-attack`/`greedy` +0.03–0.04) nets to zero.
Versus `azlite` it ticks 0.237 → 0.270, but that is inside the ±0.055 noise at this
`n` and still a ~73% loss — it did **not** learn to beat `azlite`. (Sanity check:
base vs `azlite` 0.237 reproduces the full-roster 0.24.)

## Why — and a sharper reason than before

The structural ceiling again (`docs/ppo-review.md` §8): a reactive net can't plan,
and the training opponent doesn't change that. Two compounding causes — the
all-loss signal vs a crushing opponent gives almost no gradient toward winning, and
a reactive policy can't specialize against an **adaptive searcher**. And now the
sharper reason from `docs/searchers-fiasco.md`: **`azlite` cheats with perfect
foresight** — it plans knowing every future draw, so "train against `azlite`" is
training a reactive net to beat an impossible information edge. It cannot, even in
principle, learn that reactively. A dead end, as predicted.

## Reproduce

```bash
# Search policies can now be TRAINING opponents (BattleEnv passes the forward-model
# state). Warm-start the zoo model and continue vs azlite for 200k steps:
python - <<'PY'
from sb3_contrib import MaskablePPO
from locma.envs.training import _build_env
env = _build_env("azlite:100", 0, 8, both_seat=True)
m = MaskablePPO.load("runs/ppo-shuffled-pool.zip", env=env, seed=0)
m.learn(total_timesteps=200_000, reset_num_timesteps=False)
m.save("runs/ppo-zoo-azlite.zip")
PY
for OPP in random greedy scripted max-guard max-attack azlite:100; do
  uv run locma play ppo:runs/ppo-zoo-azlite.zip $OPP --games 150 --seed 0
done
```

(~95 min at ~35 fps for the 200k `azlite` phase — the search opponent is the cost.
Wrap the training in `if __name__ == "__main__":`; `SubprocVecEnv` uses spawn.)

---

# Baselines — 2026-06-26: shuffled draft pool replaced uniform (prior matrix frozen)

_Date: 2026-06-26_

The new `ShuffledPoolSource` (PR #31) became the default draft source, replacing
the inline uniform-with-replacement sampling in `start_draft`. The pool is now
the whole 160-card space duplicated `copies=2`, shuffled, and dealt sequentially
into triplets — every card is offered **at most twice**, an even spread with
bounded duplicates, versus the old independent-per-slot draws. The top-of-file
matrix is now the shuffled-pool reference; the **prior uniform-pool** canonical
matrix (the 2026-06-25 ADR-0003 refresh) is preserved here:

|            | random | scripted | greedy | max-guard | max-attack |
|------------|--------|----------|--------|-----------|------------|
| random     | —      | 0.01     | 0.02   | 0.01      | 0.02       |
| scripted   | 0.99   | —        | 0.55   | 0.48      | 0.59       |
| greedy     | 0.98   | 0.45     | —      | 0.43      | 0.32       |
| max-guard  | 0.99   | 0.52     | 0.57   | —         | 0.57       |
| max-attack | 0.98   | 0.41     | 0.68   | 0.43      | —          |

Ranking (uniform pool): `max-attack` (59.20 / 2843) > `max-guard` (28.37 / 1973)
> `greedy` (-0.65 / 1249) > `scripted` (-18.41 / 894) > `random` (-44.11 / 541).
The shift to the shuffled pool moved cells by 1–3 points without changing the
ordering — the baselines' relative strengths are robust to the draft
distribution. (The largest single move: `scripted` vs `max-guard` 0.48 → 0.51,
flipping that head-to-head.)

---

# Baselines — 2026-06-26: PPO retrained under the shuffled draft pool (beats all ground baselines)

_Date: 2026-06-26_

Retraining the `train-zoo` curriculum (greedy → scripted → max-guard → max-attack,
200k steps each = 800k, both-seat) under the **new shuffled draft pool** default,
then evaluating the model paired with the `balanced` draft (the `ppo:` spec) vs each
baseline (`locma play ppo:runs/ppo-shuffled-pool.zip <opp> --games 500 --seed 0`,
1000 games/cell):

| policy                                   | random | scripted | greedy | max-guard | max-attack | **avg-hard3** |
|------------------------------------------|--------|----------|--------|-----------|------------|---------------|
| **ppo** (shuffled-pool zoo + `balanced`) | 0.995  | 0.596    | 0.674  | 0.577     | 0.592      | **0.588**     |

This is the **strongest PPO in this file** — it **beats all five baselines**, every
hard cell ≥ 0.577 (avg-hard3 **0.588** vs the prior best 0.569 for the both-seat
`mixed` 800k model). The `ppo:` spec pairs the battle net with the `balanced` draft,
the draft sweep's best partner. It is the **bar** for the AlphaZero-lite work.

## Reproduce

```bash
uv run locma train-zoo --out runs/ppo-shuffled-pool.zip --seed 0
for OPP in random scripted greedy max-guard max-attack; do
  uv run locma play ppo:runs/ppo-shuffled-pool.zip $OPP --games 500 --seed 0
done
```

---

# Baselines — 2026-06-26: AlphaZero-lite — search beats every PPO

_Date: 2026-06-26_

> **Correction (2026-06-26): `azlite` cheats.** This section originally framed
> `azlite` as a *non-cheating* breakthrough — that is wrong. `azlite` runs PUCT on
> the **perfect-information forward model**: it clones the real `GameState`, so its
> search sees the opponent's hidden hand *and* both decks' future draw order
> (perfect foresight) — the same class of cheating as `mcts`. The "without
> cheating" claim below is therefore false. The genuinely fair search is `dmcts`
> (`dmcts` vs `ppo` = 0.74). Full audit: `docs/searchers-fiasco.md`.

`azlite` is **AlphaZero-lite**: PUCT-guided MCTS on the perfect-information forward
model, using existing heuristics as the `f(s) -> (policy, value)` oracle instead of a
self-play-trained net (the `docs/ppo-review.md` §8.4B "search in the loop" path, in
its simplest form). The **prior** over a node's legal actions is a 1-ply heuristic
lookahead (apply each action, score the result with the board/health leaf value,
softmax); the **value** is that same leaf value. PUCT focuses the simulations, so 100
of them already play at — and past — the cheating MCTS's level. Drop-in (`azlite:100`),
self-contained (no `[ml]`), deterministic, ~0.4 s/game. Paired with the `balanced`
draft like `ppo:`.

## Win rate vs the pool (`azlite:100`, balanced draft, `--seed 0`, 300 games/cell)

| policy         | random | scripted | greedy | max-guard | max-attack | **avg-hard3** |
|----------------|--------|----------|--------|-----------|------------|---------------|
| **azlite:100** | 1.000  | 0.677    | 0.823  | 0.757     | 0.790      | **0.741**     |

It **beats every baseline** — avg-hard3 **0.741**, versus the strongest PPO's 0.588 —
the first policy in this kit to do so without a trained net. (~~without cheating~~ —
it cheats; perfect-foresight, see the correction above.)

## Head-to-head vs the strongest policies

| matchup (`azlite:100` = A)                   | win rate A | 95% CI      | n   |
|----------------------------------------------|------------|-------------|-----|
| azlite vs `ppo:runs/ppo-shuffled-pool.zip`   | **0.760**  | 0.696–0.814 | 200 |
| azlite vs `mcts:100` (cheating perfect-info) | 0.570      | 0.501–0.637 | 200 |

azlite **beats the strongest PPO head-to-head 0.76** and is **even-to-ahead of the
cheating MCTS (0.57)** — a simple, fast drop-in that beats every PPO we have tested,
by a wide margin on the baseline sweep and head-to-head.

## Why it works (and what it means for §8.4B)

The PPO study concluded the ceiling is **structural** — a reactive net cannot plan, and
the only fix is **search at play time** (`docs/ppo-review.md` §8). `azlite` is the
cheapest possible test of that claim: keep the existing heuristic as the (prior, value)
oracle and just add PUCT search. It immediately reaches MCTS strength and laps every
PPO — confirming the thesis. The *full* §8.4B build (a self-play-trained policy/value
net replacing the heuristic oracle) remains the lever for going *beyond* the cheating
MCTS; `azlite` is its working skeleton — the PUCT loop, the forward model, and the
prior/value plumbing — with a heuristic stand-in for the net.

## Reproduce

```bash
for OPP in random scripted greedy max-guard max-attack; do
  uv run locma play azlite:100 $OPP --games 150 --seed 0
done
uv run locma play azlite:100 ppo:runs/ppo-shuffled-pool.zip --games 100 --seed 0
uv run locma play azlite:100 mcts:100 --games 100 --seed 0
```

---

# Baselines — 2026-06-25: PPO zoo (semantic action space + opponent strategy)

_Date: 2026-06-25_

After the semantic action-space fix (PR #19; see `docs/ppo-review.md`), this
study re-asks the opponent-diversity question the positional-era study answered
"no" to. Three `MaskablePPO` models, **800k steps each**, differing only in
training opponent strategy:

- **single** — 800k vs `greedy` (one fixed opponent).
- **mixed** — 800k vs the `mixed` per-episode baseline pool.
- **curriculum** — 200k each, **back-to-back** on one model:
  `greedy → scripted → max-guard → max-attack` (the new `locma train-zoo`).

## Tournament (5 baselines + 3 PPO, 200 games/pair, `--seed 0`)

Pair-score matrix, row win rate vs column (PPO models abbreviated p:single /
p:mixed / p:curr):

|            | random | scripted | greedy | max-guard | max-attack | p:single | p:mixed | p:curr |
|------------|--------|----------|--------|-----------|------------|----------|---------|--------|
| scripted   | 0.98   | —        | 0.55   | 0.51      | 0.56       | 0.72     | 0.61    | 0.60   |
| greedy     | 0.98   | 0.46     | —      | 0.47      | 0.30       | 0.35     | 0.42    | 0.42   |
| max-guard  | 1.00   | 0.49     | 0.53   | —         | 0.60       | 0.70     | 0.58    | 0.64   |
| max-attack | 0.98   | 0.43     | 0.69   | 0.40      | —          | 0.65     | 0.59    | 0.66   |
| **p:single** | 0.99 | 0.28     | 0.65   | 0.30      | 0.35       | —        | 0.48    | 0.54   |
| **p:mixed**  | 0.98 | 0.39     | 0.57   | 0.41      | 0.41       | 0.52     | —       | 0.52   |
| **p:curr**   | 0.98 | 0.40     | 0.58   | 0.36      | 0.34       | 0.47     | 0.48    | —      |

Avg win rate vs the four non-`random` baselines: **p:mixed 0.445 > p:curr 0.420 >
p:single 0.395**.

## Findings

1. **Opponent diversity now helps** — the opposite of the positional-era study.
   `mixed` (0.445) and `curriculum` (0.420) both beat single-`greedy` (0.395) on
   the *hard* baselines: `single` is a `greedy`-specialist (greedy 0.65 but
   scripted 0.28 / max-guard 0.30), while `mixed` is balanced (~0.39–0.41 across
   all four ground baselines). The fixed action space unlocked a lever the broken
   one couldn't use.
2. **`mixed` is the best strategy** — it beats both other PPOs head-to-head (0.52
   vs each) and is the most balanced. `curriculum` helps over `single` but is
   slightly behind `mixed` on the hardest baselines.
3. **Still short of the ground baselines** — even `mixed` loses to `scripted`
   (0.61), `max-guard` (0.58), and `max-attack` (0.59). But this is a large step
   up from the positional era (where PPO lost ~0.70–0.77 to these); the gap is now
   ~0.58–0.61, not ~0.77.
4. **Ratings mislead again — read the matrix.** openskill/elo rank the PPOs #1/#2/#3
   (curriculum 51.3 / mixed 37.2 / single 24.6) *above every baseline* — yet
   `curriculum` (rated #1) **loses head-to-head to both other PPOs** (0.47, 0.48)
   and loses 0.58–0.66 to the ground baselines. The pair-score matrix is the truth.

**Takeaway.** Under the semantic action space, opponent diversity is a real lever
(`mixed` best, ~0.45 avg vs ground baselines), but no PPO yet *beats* the ground
baselines. Closing that gap needs the structural levers in `docs/ppo-review.md`
§8 (reward shaping, both-seat training, longer horizon), not more opponents.

## Reproduce

```bash
uv sync --extra ml
uv run locma train --steps 800000 --opponent greedy --out runs/zoo-single.zip --seed 0
uv run locma train --steps 800000 --opponent mixed  --out runs/zoo-mixed.zip  --seed 0
uv run locma train-zoo --steps-per-opponent 200000  --out runs/zoo-curriculum.zip --seed 0
uv run locma tournament random scripted greedy max-guard max-attack \
  ppo:runs/zoo-single.zip ppo:runs/zoo-mixed.zip ppo:runs/zoo-curriculum.zip \
  --games 100 --seed 0 --matrix
```

---

# Baselines — 2026-06-26: PPO is not under-capacity (net size × seat 2×2)

_Date: 2026-06-26_

Is the PPO's ceiling a mundane bug (MLP too small, single-side training, action
mapping)? A clean 2×2 — net {64×64, 256×256×256} × seat {seat-0 only, both} —
trained vs `mixed` 400k, evaluated paired with `balanced` (120 games/cell):

| net | seat | scripted | max-guard | max-attack | **avg-hard3** | vs mcts:100 |
|-----|------|----------|-----------|------------|---------------|-------------|
| 64×64 | seat-0 (shipped config) | 0.483 | 0.467 | 0.525 | 0.492 | 0.200 |
| 64×64 | **both** | 0.500 | 0.542 | 0.608 | **0.550** | 0.192 |
| 256³ | seat-0 | 0.492 | 0.392 | 0.467 | 0.450 | 0.133 |
| 256³ | both | 0.500 | 0.392 | 0.483 | 0.458 | 0.200 |

- **MLP size is not the cap — bigger *hurts*.** 256×256×256 < 64×64 at this budget
  (under-trained for the data). The SB3 default size is right.
- **Both-seat training helps the small net (+0.06, 0.49 → 0.55)** — mostly a **2×
  efficiency** win (0.55 ≈ the shipped 800k-*seat-0* model's 0.556) plus the
  *correct* thing (eval is mirrored across seats). Now the default in
  `train` / `train-zoo` (`--both-seat`).
- **Action mapping is sound** — BC-of-`greedy` reaches 0.95 agreement, so the net
  can represent and execute a strong policy from this obs/action space.
- **vs `mcts:100` flat at ~0.2 across all four** — no mundane fix closes the search
  gap, which is what makes the structural conclusion (reactive nets can't plan)
  trustworthy rather than premature.

**Full-budget confirm.** Retraining `mixed` 800k *with* both-seat reaches avg-hard3
**0.569** vs the seat-0 800k model's **0.554** (160 games/cell, paired `balanced`) —
marginally better on the ground baselines (max-attack 0.575→0.619), same ceiling, a
touch noisier vs `mcts` (0.23→0.17, run variance). Both-seat is now the canonical
`zoo-mixed` model — confirming the 2×2: at full budget the seat-0 model catches up,
so both-seat's real value is the 2× efficiency + correctness, not a higher ceiling.

---

# Baselines — 2026-06-25: MCTS heuristic rollout (turn-based, new default)

_Date: 2026-06-25_

MCTS rollouts are now **heuristic** by default (`rollout_turns=3`): random-play a
few *turn boundaries* (adaptive depth — a turn is variable‑length), then score the
settled position with a board/health heuristic, instead of random‑playing to
terminal. Random rollouts are high‑variance and weak at 100 iterations; the
heuristic leaf value is both much **stronger** and ~6× faster. Combined with the
fast battle clone, `mcts:100` went from ~7.8 s/game → **~0.2 s/game (~30×)**. The
legacy terminal rollout is `rollout_turns<=0` (spec `mcts:100,1.41,0,0`).

## `mcts:100` (heuristic) win rate vs the pool

`locma play mcts:100 <opp> --games 60 --seed 0` (120 games/cell):

| vs → | random | scripted | greedy | max‑guard | max‑attack | ppo (balanced) |
|------|--------|----------|--------|-----------|------------|----------------|
| **mcts:100** | 1.000 | 0.658 | 0.908 | 0.767 | 0.708 | **0.733** |

It is now the **strongest policy in the kit by a clear margin** — it beats every
baseline and the new PPO (which the *old* random‑rollout `mcts:100` was only
~even with). The old `mcts:100` beat `greedy` ~0.79; that section below is
superseded by this default.

## Reproduce

```bash
uv run locma play mcts:100 greedy --games 60 --seed 0            # heuristic (default)
uv run locma play mcts:100,1.41,0,0 greedy --games 60 --seed 0   # legacy terminal rollout
```

---

# Baselines — 2026-06-25: PPO × draft sweep (the deck is the lever)

_Date: 2026-06-25_

Follow-up to `docs/ppo-review.md` §8.1 (the gap between PPO and the ground
baselines is mostly the DECK). This sweep pairs the trained PPO battle net with
every draft heuristic — including three new ones, `max-defense` / `balanced` /
`weighted` (`locma/policies/drafts.py`) — and asks two things. **avg-hard3** =
mean win rate vs the three hard baselines (`scripted` / `max-guard` /
`max-attack`); 300 games/cell, held-out seeds (`1_000_000+`).

## (B) Pair the mixed-trained PPO battle net with each draft (eval only)

| draft (+ PPO battle) | scripted | max-guard | max-attack | **avg-hard3** |
|----------------------|----------|-----------|------------|---------------|
| **max-guard**        | 0.553    | 0.527     | 0.567      | **0.549** |
| **balanced** (new)   | 0.507    | 0.540     | 0.587      | **0.544** |
| random               | 0.460    | 0.453     | 0.557      | 0.490 |
| max-attack           | 0.393    | 0.447     | 0.527      | 0.456 |
| weighted (new)       | 0.397    | 0.433     | 0.450      | 0.427 |
| max-defense (new)    | 0.397    | 0.380     | 0.440      | 0.406 |
| **greedy** (shipped) | 0.347    | 0.410     | 0.423      | **0.393** |

## (C) Train the PPO battle net with the AGENT drafting each heuristic (300k)

| draft | random | greedy | max-guard | max-attack | max-defense | balanced | weighted |
|-------|--------|--------|-----------|------------|-------------|----------|----------|
| **(C) trained avg-hard3** | 0.454 | 0.407 | 0.543 | 0.467 | 0.398 | 0.528 | 0.404 |
| **(B) paired avg-hard3**  | 0.490 | 0.393 | 0.549 | 0.456 | 0.406 | 0.544 | 0.427 |

## Findings

1. **The draft dominates, and `greedy` (the shipped draft) is the *worst* of all
   seven** (0.393) — even a **random** draft (0.490) beats it. `max-guard` (0.549)
   and the new `balanced` (0.544) are best, and both make the PPO **beat all three
   ground baselines** (every cell ≥ 0.50) — purely by swapping the draft, no
   retraining.
2. **Training-on-deck ≈ just-pairing** — (C) and (B) match within noise for every
   draft. The battle net is **deck-robust**: the deck at *deployment* is the
   lever, not the deck it trained on. So there is no extra gain from retraining
   per draft; choose the deck.
3. **New heuristics:** `balanced` ties `max-guard` for best (curve + creature
   majority + Guard value). `weighted` (0.43) and `max-defense` (0.41) underperform
   — raw stats / keyword value without the Guard-wall + curve structure don't help.

**Actionable:** pair the `ppo:` policy with a `max-guard` or `balanced` draft
instead of `greedy` — that alone turns it from losing (~0.39) to **beating** the
ground baselines (~0.55), with no retraining.

## Spell-aware draft valuation (refinement)

Item cards carry stats applied to the **enemy**: red/blue removal spells have
**negative** attack/defense (e.g. *Decimate* defense −99 = destroy a minion,
*Mighty Throwing Axe* defense −7 = 7 damage). The stat-summing heuristics scored
these by `attack + defense`, valuing *Decimate* at **−99** — the worst card in the
game. Fixed: `_card_value` (`locma/policies/drafts.py`) now values items by the
**magnitude** of their effect (`|attack| + |defense|`, capped at 13 so
destroy-sentinels don't dominate) + keyword value.

But *correct* spell valuation **hurt** the PPO pairing — the learned battle net
plays creatures far better than spells (`ppo+balanced` 0.544 → 0.487 once it drafted
removal). Tuning the `balanced` item discount against the PPO net (1.5 → 6 → 12 gave
0.47 → 0.52 → 0.56 avg vs the hard baselines) settled on a strong creature bias: the
shipped `balanced` drafts creature-heavy and takes only premium removal, reaching
**0.556** (scripted 0.520 / max-guard 0.553 / max-attack 0.593 — beats all three),
the best draft in the sweep. `greedy` is left deliberately naive as the reference
baseline.

This is **not** a training-distribution gap: training a PPO battle net *on*
spell-heavy hands (≈1.1 removal/deck) did not improve spell-deck play (0.493 vs
0.487 for the creature-trained net) and made it worse overall (0.492 vs 0.556 on
the good deck). Spell decks are simply weaker for this aggressive tempo style — a
creature gives recurring board presence and face damage; one-shot removal trades
1-for-1 without advancing your own clock.

## Reproduce

```bash
# (B) pair the trained battle net with each draft (eval only)
#   Composer(MaskablePPOBattlePolicy('runs/zoo-mixed.zip'), <Draft>Policy())  vs baselines
# (C) train with the agent drafting each heuristic, then eval — see the prototype harness.
```

---

# Baselines — 2026-06-25: MCTS distillation (practicum → BC)

_Date: 2026-06-25_

Can a **fast reactive net** distill the strength of the slow **cheating**
`mcts:100` search? We recorded a *practicum* — 45,590 `(observation, expert
action, mask)` examples from `mcts:100` playing both seats vs each baseline
(156 games/opponent × 2 seats, 1,556 games, 0 dropped/failed) — and
behavior-cloned it into a `MaskablePPO` net via masked cross-entropy
(`record-practicum` → `distill`). The student sees only the imperfect
`BattleView` obs; the teacher cheats (perfect information). Pipeline and
commands are reproducible; the practicum/model artifacts live in gitignored
`runs/`.

## Result 1 — top-1 agreement plateaus at ~0.25 (information gap, not underfit)

Held-out **game-level** split (10% of games). The net's masked argmax matches
`mcts:100` on only **~25%** of decisions (random over ~21 legal actions ≈ 5%, so
it learns *something*, but it is nowhere near greedy's 0.78 in the same
pipeline). This is the **teacher–student information gap**: raising the learning
rate and quadrupling epochs drives the **training** loss down (1.74 → 1.61) while
**val agreement stays flat at 0.25** — the student's observation simply does not
determine a perfect-information searcher's move (the teacher acts on the
opponent's hidden hand and deep lookahead the obs never exposes).

| distill config        | train loss (final) | val top-1 agreement |
|-----------------------|--------------------|---------------------|
| 10 epochs, lr 3e-4    | 1.712              | 0.255               |
| 40 epochs, lr 1e-3    | 1.609              | 0.250               |

## Result 2 — the distilled net plays at *from-scratch-PPO* level, not teacher level

Win rate of the distilled net (10-epoch model) vs each baseline (100 games/cell,
mirrored, `--seed 0`), beside the from-scratch PPO (100k vs `greedy`) row from the
section below and the cheating teacher:

| vs →            | random | greedy | scripted | max-guard | max-attack |
|-----------------|--------|--------|----------|-----------|------------|
| **distilled** (BC) | 0.960 | 0.480 | 0.340 | 0.285 | 0.250 |
| from-scratch PPO   | 0.974 | 0.504 | 0.292 | 0.275 | 0.228 |
| `mcts:100` (teacher) | ~0.98 | **~0.79** | strong | strong | strong |

The distilled profile is **the from-scratch-PPO profile** — crushes `random`,
even-ish with `greedy`, **loses 0.25–0.34 to the ground baselines**. It inherits
**none** of the teacher's 0.79-vs-`greedy` edge. Worse, **more training hurts
strength**: the 40-epoch model drops to greedy 0.41 / scripted 0.225 / max-guard
0.195 / max-attack 0.17 — extra epochs overfit the *marginal* action distribution
(imitating common moves) rather than learning to win.

## Result 3 — speed is the only thing that transferred (and it's moot)

The distilled net is **~58× faster** than the teacher (`mcts:100` ≈ 7.2 s/game vs
distilled ≈ 0.13 s/game). But a fast net that plays at PPO strength is just
PPO — the speed buys nothing the baselines didn't already have.

## Verdict

**You cannot distill a cheating, perfect-information search into an
imperfect-information reactive net by behavior cloning.** The bottleneck is the
**observation**, not the training method: the student is blind to exactly the
information (opponent's hidden hand + lookahead) that makes `mcts:100` strong, so
its best imitation collapses to the generic PPO policy. This reinforces the PPO
study's conclusion — the ceiling is **structural** (observation / reward /
battle-only training), not a matter of training effort or a better teacher. The
practicum/distill machinery works and is reusable; the next lever is a **richer
observation or reward**, or distilling a **non-cheating** search (one limited to
the same `BattleView` the student sees) whose decisions the student *can* learn.

### Reproduce

```bash
# generate (per-opponent shards, concurrent) then merge -> runs/practicum.npz
for OPP in random scripted greedy max-guard max-attack; do
  uv run locma record-practicum --teacher mcts:100 --opponents $OPP \
    --games 156 --out runs/practicum-$OPP.npz --seed 0 &
done; wait   # then concatenate shards (offset game_id, set opponent_id)
uv run locma distill --data runs/practicum.npz --out runs/distilled.zip \
  --epochs 10 --batch 256 --lr 3e-4 --val-frac 0.1 --seed 0
for OPP in random scripted greedy max-guard max-attack; do
  uv run locma play ppo:runs/distilled.zip $OPP --games 100 --seed 0
done
```

---

# Baselines — 2026-06-25 (cont.): PPO budget × opponent study

_Date: 2026-06-25_

The single 100k PPO model below overfit its trainer and lost to the ground
baselines. This study asks whether **more budget** or **opponent diversity**
breaks that ceiling. Three PPO trajectories — trained against `greedy` (single),
`mixed` (a new per-episode pool of all five baselines, the `mixed` preset), and
`max-attack` (the toughest baseline) — were each trained as **one continuous,
seeded, checkpointed run** to 3M steps, saving models at 100k / 300k / 1M / 3M
(the 1M model *is* the 3M run at step 1M). Each checkpoint was evaluated vs all
five baselines (200 games/cell). Training is now seeded (was not — see the
methodology note) and ran three trajectories concurrently for CPU use.

## Does budget help? (PPO win rate vs each baseline, by training steps)

Each block is one training opponent; rows are the budget; cells are the PPO
checkpoint's win rate (200 games). **The curves are flat** — 30× more compute
does not improve play against the ground baselines.

**Trained vs `greedy`:**

| steps | random | scripted | greedy | max-guard | max-attack |
|-------|--------|----------|--------|-----------|------------|
| 100k  | 0.970  | 0.325    | 0.490  | 0.265     | 0.215      |
| 300k  | 0.980  | 0.275    | 0.440  | 0.275     | 0.205      |
| 1M    | 0.980  | 0.230    | 0.445  | 0.285     | 0.195      |
| 3M    | 0.985  | 0.260    | 0.455  | 0.280     | 0.240      |

**Trained vs `mixed` (pool of all five baselines):**

| steps | random | scripted | greedy | max-guard | max-attack |
|-------|--------|----------|--------|-----------|------------|
| 100k  | 0.965  | 0.305    | 0.440  | 0.265     | 0.230      |
| 300k  | 0.985  | 0.390    | 0.470  | 0.305     | 0.250      |
| 1M    | 0.985  | 0.285    | 0.365  | 0.275     | 0.240      |
| 3M    | 0.995  | 0.290    | 0.460  | 0.250     | 0.230      |

**Trained vs `max-attack` (toughest baseline):**

| steps | random | scripted | greedy | max-guard | max-attack |
|-------|--------|----------|--------|-----------|------------|
| 100k  | 0.985  | 0.260    | 0.410  | 0.320     | 0.215      |
| 300k  | 0.980  | 0.310    | 0.425  | 0.320     | 0.245      |
| 1M    | 0.980  | 0.300    | 0.480  | 0.330     | 0.295      |
| 3M    | 0.970  | 0.290    | 0.430  | 0.300     | 0.240      |

Average win rate vs the four non-`random` baselines, by budget (trained vs
greedy): 0.32 → 0.30 → 0.29 → 0.31. **Flat to slightly down.** Budget is not the
bottleneck.

## Does opponent diversity help? (at 3M steps)

Average win rate vs the four non-`random` baselines, per training opponent:
`greedy`-trained 0.31, `mixed`-trained 0.31, `max-attack`-trained 0.31 —
**identical**. The `mixed` generalist did not generalize better than the single
opponent; `max-attack`-trained is marginally best vs `max-guard` (it sees the
ground style) but still loses. Every PPO variant, at every budget, **crushes
`random` (~0.97–0.99), is roughly even-to-below `greedy` (~0.37–0.49), and loses
to `scripted`, `max-guard`, and `max-attack`** (~0.20–0.33). The profile never
changes.

## Full policy matrix (baselines + the three PPO at 3M)

`locma tournament random scripted greedy max-guard max-attack ppo:...greedy-3M
ppo:...mixed-3M ppo:...max-attack-3M --games 500 --seed 0 --matrix`. Columns
`pG`/`pX`/`pA` = PPO trained vs greedy / mixed / max-attack. Row = win rate vs
column:

|            | random | scripted | greedy | max-guard | max-attack | pG   | pX   | pA   |
|------------|--------|----------|--------|-----------|------------|------|------|------|
| random     | —      | 0.01     | 0.02   | 0.01      | 0.02       | 0.03 | 0.02 | 0.04 |
| scripted   | 0.99   | —        | 0.55   | 0.48      | 0.59       | 0.74 | 0.70 | 0.71 |
| greedy     | 0.98   | 0.45     | —      | 0.43      | 0.32       | 0.48 | 0.56 | 0.57 |
| max-guard  | 0.99   | 0.52     | 0.57   | —         | 0.57       | 0.73 | 0.75 | 0.73 |
| max-attack | 0.98   | 0.41     | 0.68   | 0.43      | —          | 0.77 | 0.77 | 0.76 |
| pG (greedy)| 0.97   | 0.26     | 0.52   | 0.27      | 0.23       | —    | 0.52 | 0.53 |
| pX (mixed) | 0.98   | 0.30     | 0.44   | 0.25      | 0.23       | 0.48 | —    | 0.51 |
| pA (max-at)| 0.96   | 0.29     | 0.43   | 0.27      | 0.24       | 0.47 | 0.49 | —    |

The three ground baselines (`scripted`, `max-guard`, `max-attack`) **beat every
PPO variant 0.70–0.77**. PPO only dominates `random` and is even-ish with
`greedy`.

**Ratings mislead, again and harder.** At every budget the tournament ranks the
three PPO models **#1/#2/#3** (3M: openskill 68.8 / 37.0 / 11.0, Elo 3356 / 2461
/ 1769) *above all baselines* — while each loses head-to-head to three of the
five. `max-attack`-trained is rated #1 yet loses 0.77 to `max-attack` itself. The
latent-skill models are systematically fooled by the "annihilate `random`, beat
the other PPOs, near-even `greedy`" profile. **Read the matrix.**

## Takeaway

Neither **30× more budget** (100k → 3M) nor **opponent diversity** (single,
pooled, or toughest) lets this PPO setup beat the ground baselines — the ceiling
is structural, not a training-time problem. Likely causes: a **flat MLP
observation** (no board/relational structure), a **sparse win/loss reward** (no
shaping), and **battle-only training** (the opponent drafts both decks, so PPO
never learns draft and is deployed on greedy-drafted decks it did not choose).
For reference, the cheating MCTS already beats `greedy` 0.79 with zero training —
search dominates this learned setup. Next levers worth trying are reward shaping,
a structured observation, or self-play/league (deferred), not more steps.

## Methodology note (reproducibility)

Training is now **seeded** (`MaskablePPO(..., seed=seed)`); the earlier
single-100k and per-opponent-checkpoint numbers (sections below) were from an
**unseeded** trainer and are representative rather than bit-reproducible. The
`--checkpoints` flag trains one continuous trajectory and saves a step-suffixed
model at each mark, so a budget point is reproducible without retraining. Models
live in gitignored `runs/`.

## Reproduce

```bash
uv sync --extra ml
# three seeded checkpointed trajectories (run concurrently for CPU use)
for OPP in greedy mixed max-attack; do
  uv run locma train --opponent $OPP --seed 0 \
    --checkpoints 100000,300000,1000000,3000000 --out runs/ppo-$OPP.zip &
done; wait
# evaluate any checkpoint, e.g. the mixed-3M generalist vs the ground baselines
uv run locma play ppo:runs/ppo-mixed-3000000.zip max-attack --games 100 --seed 0
# full policy matrix at 3M
uv run locma tournament random scripted greedy max-guard max-attack \
  ppo:runs/ppo-greedy-3000000.zip ppo:runs/ppo-mixed-3000000.zip \
  ppo:runs/ppo-max-attack-3000000.zip --games 500 --seed 0 --matrix
```

---

# Baselines — 2026-06-25: MCTS + PPO (post-split refresh)

_Date: 2026-06-25_

Adds two new policy families — a search policy (`mcts`) and a learned policy
(`ppo`) — on top of the policy-split refactor (draft/battle halves recombined by
a `Composer`; see `CONTEXT.md` and `docs/adr/`). The refactor gave the two
**stochastic** baselines (`random`, `scripted`) independent per-half RNGs (ADR
0003), shifting their win rates 1–3 points; the deterministic baselines
(`greedy`, `max-guard`, `max-attack`) are byte-identical to before. The 5-policy
matrix at the top of this file is the refreshed reference. All runs `--seed 0`.

## New policies

- **`mcts:iterations,c,seed`** — cheating perfect-information MCTS (UCT) battle
  policy paired with a greedy draft. Uses the engine as a forward model
  (deep-copy + simulate ahead); random rollouts to terminal. `mcts:100` = 100
  simulations per decision. ~7 s/game, so it is evaluated at smaller n than the
  deterministic baselines.
- **`ppo:model_path`** — MaskablePPO battle policy paired with a greedy draft.
  Battle-only training (the opponent drafts both decks). Trained models are
  gitignored (`runs/`); the numbers below need a local `locma train` (and the
  `[ml]` extra) to reproduce.

## MCTS (`mcts:100`, greedy draft)

`locma play mcts:100 <opp> --games 50 --seed 0` (100 games):

| Matchup            | Win rate | 95% CI      | n   |
|--------------------|----------|-------------|-----|
| mcts:100 vs random | 1.000    | 0.963–1.000 | 100 |
| mcts:100 vs greedy | 0.790    | 0.700–0.858 | 100 |

`mcts:100` is the **strongest battle policy in the kit**: perfect vs `random`,
beats `greedy` 79%. Search depth is load-bearing — a shallow `mcts:30` is much
weaker. Both sides draft greedily here, so this isolates battle strength.

## PPO (100k steps vs `greedy`)

`ppo:runs/ppo-greedy-100k.zip`, `locma play ppo <opp> --games 500 --seed 0`
(1000 games):

| Matchup           | Win rate | 95% CI      | binomial p |
|-------------------|----------|-------------|------------|
| ppo vs random     | 0.974    | 0.962–0.982 | 3.4e-250   |
| ppo vs greedy     | 0.504    | 0.473–0.535 | 0.82       |
| ppo vs scripted   | 0.292    | 0.265–0.321 | 1.7e-40    |
| ppo vs max-guard  | 0.275    | 0.248–0.303 | 2.3e-47    |
| ppo vs max-attack | 0.228    | 0.203–0.255 | 1.1e-69    |

Trained against `greedy`, PPO learned to **match greedy** (0.50) and **crush
random** (0.97), but **overfit to its trainer**: it loses to every other style.

**Rating caveat (read the matrix).** In a 6-policy tournament including PPO, the
openskill/Elo models rank PPO **#1** (openskill 60.3, Elo 2956) — yet PPO loses
head-to-head to three of the five baselines. The rating models are fooled by
PPO's profile (annihilates `random`, even with `greedy`); the pair-score matrix
is the truth. Same non-transitivity caveat as the 5-policy pool, amplified.

## PPO checkpoint study — does the training opponent matter?

Five PPO checkpoints, one per baseline opponent (100k steps each), each
evaluated against all five baselines.
`locma play ppo:runs/ppo-<train>-100k.zip <eval> --games 100 --seed 0`
(200 games/cell, CI half-width ≈ ±0.07; cell = PPO win rate):

| train \ eval | random | scripted | greedy | max-guard | max-attack |
|--------------|--------|----------|--------|-----------|------------|
| random       | 0.975  | 0.310    | 0.475  | 0.325     | 0.220      |
| scripted     | 0.945  | 0.225    | 0.340  | 0.240     | 0.150      |
| greedy       | 0.980  | 0.275    | 0.480  | 0.285     | 0.240      |
| max-guard    | 0.975  | 0.295    | 0.455  | 0.285     | 0.220      |
| max-attack   | 0.975  | 0.330    | 0.470  | 0.305     | 0.230      |

**The training opponent barely matters at this budget.** Three patterns:

1. **Eval opponent dominates, not training opponent.** Every checkpoint crushes
   `random` (~0.95–0.98) and loses to `scripted`/`max-guard`/`max-attack`
   (~0.15–0.33) regardless of who it trained against — the rows are nearly
   identical, the columns vary.
2. **No exploitation on the diagonal.** Training against X does *not* make PPO
   beat X — `scripted`-trained vs `scripted` = 0.225 (its worst cell),
   `max-attack`-trained vs `max-attack` = 0.230. The agent trained against a
   strong opponent mostly just gets crushed and learns little transferable.
3. **Best generalists are marginal.** Averaged over the four non-`random`
   baselines: `max-attack`-trained (0.334) ≈ `random`-trained (0.333) >
   `greedy` (0.320) > `max-guard` (0.314) > `scripted`-trained (0.239) — spread
   within the n=200 noise.

**Takeaway.** Per-opponent training does not produce specialists here; the
bottleneck is the *learning* (100k steps, flat MLP obs, sparse win/loss reward,
battle-only env), not opponent diversity. A mixed-opponent pool or self-play
might raise the floor, but more steps, a shaped reward, or a richer observation
are the likelier levers. (Self-play is deferred — see the plan.)

## Reproduce

```bash
# MCTS (cheating perfect-info; ~7 s/game at 100 sims)
uv run locma play mcts:100 random --games 50 --seed 0
uv run locma play mcts:100 greedy --games 50 --seed 0

# PPO — train (requires the [ml] extra); models land in gitignored runs/
uv sync --extra ml
for OPP in random scripted greedy max-guard max-attack; do
  uv run locma train --steps 100000 --opponent $OPP --out runs/ppo-$OPP-100k.zip --seed 0
done

# PPO single (greedy checkpoint) + 6-policy tournament
uv run locma play ppo:runs/ppo-greedy-100k.zip greedy --games 500 --seed 0
uv run locma tournament random scripted greedy max-guard max-attack \
  ppo:runs/ppo-greedy-100k.zip --games 500 --seed 0 --matrix

# PPO checkpoint train×eval matrix (200 games/cell)
for TR in random scripted greedy max-guard max-attack; do
  for EV in random scripted greedy max-guard max-attack; do
    uv run locma play ppo:runs/ppo-$TR-100k.zip $EV --games 100 --seed 0
  done
done
```

## Takeaways

- **`mcts:100` is the new strongest policy** — beats `greedy` 0.79, perfect vs
  `random`. Cheating perfect-info + enough search wins; shallow search does not.
- **PPO (100k vs greedy) is a `greedy`/`random` specialist** — matches its
  trainer, crushes random, loses to the ground baselines.
- **Training opponent ≈ irrelevant at this budget** — the five PPO checkpoints
  are nearly interchangeable; the learning setup, not opponent choice, is the
  limit.
- **Ratings mislead for PPO** — it rates #1 while losing to 3/5 baselines; the
  pair-score matrix is authoritative.

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

## PPO2 — tokenized observation + self-attention (richer-encoding lever)

`obs_mode="token"` adds a tokenized board encoding — per-card tokens (zone/type/cost/
attack/defense/abilities/readiness + a learned **card-id** embedding the flat obs
discards) + computed tactical scalars (guard count, reachable face damage, lethal,
mana, board totals) — consumed by a **slot-addressable self-attention** extractor
(`TokenSetExtractor`) and trained via `MultiInputPolicy`. It is additive:
`obs_mode="flat"` (default) is byte-identical to the prior baseline and is the A/B
control. The token arm uses gentler PPO (`--learning-rate 1e-4 --target-kl 0.025`);
the flat arm keeps the defaults (the bigger net is unstable at 3e-4 — `approx_kl`
blows up and it degrades with training).

**A/B (zoo curriculum greedy→scripted→max-guard→max-attack, 200k×4 = 800k/arm, 2
seeds, eval 400 games/opp at seed 0):**

| arm   | seed 0 | seed 1 | mean avg-hard3 | mean vs greedy |
|-------|--------|--------|----------------|----------------|
| flat  | 0.592  | 0.555  | **0.573**      | 0.639          |
| token | 0.562  | 0.614  | **0.588**      | 0.664          |

Token edges flat on the 2-seed mean (avg-hard3 +0.015, greedy +0.025), but the **seeds
disagree** (seed 0 → flat wins, seed 1 → token wins) and the per-seed spread
(~0.03–0.04) exceeds the gap, so the win is **not significant at n=2**. Single-opponent
training (300k vs max-attack) favors flat (0.588 vs token 0.565) because the larger
token net overfits the lone opponent. **Verdict: parity / a slight lean toward token
under the curriculum — a secondary lever, within seed noise.** Token costs ~4× the
training wall-clock. Full analysis in `ppo-review.md` §8.4A.

Reproduce:
```bash
# token arm (one seed)
uv run locma train-zoo --steps-per-opponent 200000 --obs-mode token \
  --learning-rate 1e-4 --target-kl 0.025 --seed 0 --out runs/ppo2-token.zip
# flat baseline (one seed)
uv run locma train-zoo --steps-per-opponent 200000 --obs-mode flat \
  --seed 0 --out runs/ppo2-flat.zip
# eval avg-hard3 (token model auto-detected via the loaded obs space)
uv run locma tournament ppo:runs/ppo2-token.zip scripted max-guard max-attack \
  --games 200 --seed 0 --matrix
```

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
