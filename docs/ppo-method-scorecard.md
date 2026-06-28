# PPO method scorecard

_Date: 2026-06-27_

Goal: rank fair/no-search ways to augment PPO so the deployed policy remains a
single MaskablePPO-style forward pass. Search wrappers (`puct-ppo`, `dpuct-ppo`)
are diagnostics only, not target methods.

Scores are 0-5:

- **WR upside**: observed or plausible win-rate improvement.
- **Fairness**: whether the method preserves a fair visible-state policy.
- **Cost**: training/eval cost and implementation complexity, inverted so higher
  is better.
- **Robustness**: consistency across hard scripted baselines and `dmcts`.

## Hypothesis

The best observed no-search improvements come from conservative PPO continuation
and richer fair opponent mixtures, not scalar reward shaping. These methods
improve tempo/face-pressure decisions and lethal taking, but they do not fix the
largest remaining tactical gap: PPO still under-uses items and sees fewer lethal
opportunities than `dmcts`. Closing the fair-search gap probably needs either an
action/representation change that makes item and target interactions easier, or a
stronger visible-information teacher that specifically labels item/guard/lethal
tactics.

## Method scores

| method | score | evidence | next action |
|--------|------:|----------|-------------|
| Sparse PPO continuation tuning | 3.5 | Best direct no-search gains so far. `lr1e-4 ent0.01 300k` beat restored head-to-head 0.540 and improved hard3 in larger eval. `lr2e-4 ent0.005 100k` preserved the best small `dmcts` probe. | Keep as current baseline challenger; validate on larger held-out matrix before promoting. |
| Rich fair opponent mixture | 3.0 | `mixed-rich` improved hard baselines and was competitive with best tuned sparse PPO. Search-heavy/laddder follow-ups gave small avg-hard3 bumps but no robust head-to-head or default-`dmcts` gain. | Keep as a controlled curriculum ingredient; do not promote without a targeted item/lethal diagnostic improvement. |
| Focused heuristic-opponent continuation | 2.5 | Small hard-baseline gains, but target-specific training did not reliably improve the target matchup. | Use only as short curriculum phase inside broader mix, not as a standalone path. |
| Dense scalar rewards | 1.5 | Health/board/tempo shaping were flat or negative; lower reward scale did not rescue them. | Keep only the small health/board scaffold; `tempo` was pruned. |
| Tactical scalar observation | 1.5 | 200k tactical zoo scout underperformed restored PPO and worsened some lethal behavior. | Do not spend full run unless paired with architecture/teacher changes. |
| Card-token extractor | 1.5 | Default LR was unstable; lower LR stabilized but remained below restored PPO. | Pruned; main's tokenized observation/PPO2 path supersedes this scout. |
| One-ply board/health oracle cloning | 1.0 | `oracle1` was weak, especially vs scripted and max-guard. | Pruned from code; keep result as paper trail only. |
| PPO-prior search wrappers | diagnostic | Showed PPO priors help search, but cheating `puct-ppo` is not fair and fair `dpuct-ppo` is a search policy at inference. | Use only to diagnose policy priors, not as target solution. |

## Compact scorecard bench

Command:

```bash
uv run locma ppo-scorecard \
  restored=ppo:runs/ppo-shuffled-pool.zip \
  tuned=ppo:runs/ppo-deep-sparse-lr1e4-ent01-300k.zip \
  rich=ppo:runs/ppo-richmix-lr1e4-ent01-150k.zip \
  tactical=ppo-tactical:runs/ppo-tactical-zoo-200k.zip \
  reward=ppo:runs/ppo-shuffled-board-s002-100k.zip \
  --games 80 --dmcts-games 40 --seed 12000000
```

Result:

| policy | scripted | greedy | max-guard | max-attack | `dmcts` | avg hard3 |
|--------|----------|--------|-----------|------------|---------|-----------|
| restored | 0.512 | 0.650 | 0.575 | 0.588 | 0.263 | 0.604 |
| tuned sparse | 0.550 | 0.681 | 0.594 | 0.637 | 0.237 | 0.638 |
| rich mix | 0.575 | 0.725 | 0.575 | 0.637 | 0.263 | 0.646 |
| tactical obs | 0.494 | 0.588 | 0.487 | 0.537 | 0.138 | 0.537 |
| board reward | 0.512 | 0.694 | 0.550 | 0.662 | 0.275 | 0.635 |

Read: the rank order in this compact bench matches the larger experiments:
rich mix and sparse continuation are the only current positive no-search
methods. Reward shaping can produce isolated matchup gains but is less balanced.

## Rich schedule follow-up

Command:

```bash
uv run locma ppo-scorecard \
  restored=ppo:runs/ppo-shuffled-pool.zip \
  best_sparse=ppo:runs/ppo-deep-sparse-lr1e4-ent01-300k.zip \
  rich=ppo:runs/ppo-richmix-lr1e4-ent01-150k.zip \
  richsearch=ppo:runs/ppo-richsearch-lr1e4-ent01-150k.zip \
  richladder=ppo:runs/ppo-richladder-lr1e4-ent01-150k.zip \
  richschedule=ppo:runs/ppo-richschedule-lr1e4-ent01-210k.zip \
  --games 80 --dmcts-games 0 --workers 8 --seed 14000000
```

Result:

| policy | scripted | greedy | max-guard | max-attack | avg hard3 |
|--------|----------|--------|-----------|------------|-----------|
| restored | 0.581 | 0.631 | 0.562 | 0.588 | 0.594 |
| best sparse | 0.588 | 0.631 | 0.544 | 0.625 | 0.600 |
| rich | 0.575 | 0.669 | 0.588 | 0.575 | 0.610 |
| richsearch | 0.581 | 0.650 | 0.594 | 0.594 | 0.612 |
| richladder | 0.575 | 0.650 | 0.575 | 0.619 | 0.615 |
| richschedule | 0.594 | 0.669 | 0.531 | 0.613 | 0.604 |

Small default-`dmcts` probe at 100 actual games each:

| policy | `dmcts` |
|--------|---------|
| restored | 0.320 |
| rich | 0.260 |
| richsearch | 0.230 |
| richladder | 0.300 |

Head-to-head vs restored at 400 actual games: rich 0.525, richsearch 0.505,
richladder 0.487. Read: richer self-play schedules are a weak positive for hard
scripted coverage, but not enough to move the core fair-search gap.

## Repeatable bench command

`locma ppo-scorecard` accepts `label=policy_spec` entries:

```bash
uv run locma ppo-scorecard \
  base=ppo:runs/ppo-shuffled-pool.zip \
  candidate=ppo:runs/ppo-richmix-lr1e4-ent01-150k.zip \
  --games 100 --dmcts-games 50 --workers 8 --seed 13000000
```

Use `--workers` on multicore machines. Use `--dmcts-games 0` for fast
hard-opponent ranking, keep `dmcts-games` low for scout runs, then rerun the best
few candidates with full `dmcts` cells.
