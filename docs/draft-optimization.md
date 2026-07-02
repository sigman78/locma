# Draft Optimization Research

Date: 2026-06-28

Goal: make deck construction measurable, then use that measurement to search for
better draft policies and better card-cost priors. The prior PPO work showed that
deck choice is one of the few levers that moved learned policy strength, so draft
quality needs its own instrumentation rather than being inferred only from full
policy tournaments.

## Measurement Added

New command:

```bash
uv run locma draft-report draft:random draft:greedy draft:weighted \
  draft:balanced draft:weighted-balanced draft:truecost-balanced \
  draft:aggro draft:midrange draft:defense draft:max-guard \
  draft:max-attack draft:max-defense --drafts 500 --seed 0 --top-cards 12
```

Implementation: `locma/harness/deck_quality.py`.

What it measures:

- `quality`: a transparent proxy, not a learned oracle:
  `avg card value + avg effective-cost delta - 0.15 * curve_l1`.
- `card value`: full-card heuristic value using stats, abilities, HP swing,
  draw, and capped removal magnitude.
- `eff cost delta`: estimated effective mana cost minus printed mana cost, fit
  from the full 160-card pool. Positive means the card/deck looks underpriced by
  the current proxy.
- `curve L1`: distance from the current balanced curve target
  `{0:1, 1:3, 2:5, 3:5, 4:5, 5:4, 6:3, 7+:4}`.
- `G/W/L`: average Guard/Ward/Lethal creature counts per 30-card deck.

This is intentionally cheap. It does **not** replace battle tournaments. Its job
is to rank draft hypotheses before spending thousands of games on them.

## Initial 500-Draft Proxy Report

Command:

```bash
COLUMNS=220 uv run locma draft-report draft:random draft:greedy \
  draft:weighted draft:balanced draft:max-guard draft:max-attack \
  draft:max-defense --drafts 500 --seed 0 --top-cards 12
```

| policy | quality | card value | eff cost delta | avg cost | creatures | items | curve L1 | draw | G/W/L |
|--------|--------:|-----------:|---------------:|---------:|----------:|------:|---------:|-----:|-------|
| `draft:balanced` | 9.48 | 10.90 | +0.01 | 4.83 | 29.3 | 0.7 | 9.5 | 1.8 | 10.5/3.1/2.1 |
| `draft:weighted` | 9.07 | 11.98 | -0.10 | 5.39 | 27.8 | 2.2 | 18.8 | 2.3 | 9.4/3.0/1.9 |
| `draft:max-guard` | 8.91 | 10.81 | +0.02 | 4.78 | 29.4 | 0.6 | 12.8 | 1.9 | 15.4/2.9/1.7 |
| `draft:max-defense` | 8.79 | 11.35 | -0.15 | 5.18 | 29.4 | 0.6 | 16.1 | 2.4 | 9.6/2.6/2.0 |
| `draft:greedy` | 8.51 | 11.58 | -0.26 | 5.38 | 28.5 | 1.5 | 18.8 | 2.4 | 8.4/2.8/1.7 |
| `draft:max-attack` | 8.48 | 11.28 | -0.23 | 5.22 | 29.4 | 0.6 | 17.1 | 2.7 | 7.5/2.9/1.6 |
| `draft:random` | 6.67 | 8.33 | +0.00 | 3.75 | 21.8 | 8.2 | 11.1 | 3.4 | 6.4/2.0/1.8 |

Read:

- `draft:balanced` tops the cheap proxy mostly because it keeps the best curve
  while staying creature-heavy. This agrees with the existing battle-tournament
  finding that the balanced draft is the best partner for PPO-like battle nets.
- `draft:weighted` has the highest raw card value but a bad curve. This is a
  promising next target: combine its card-value model with stronger curve pressure.
- `draft:max-guard` remains competitive despite lower raw value because Guard
  density is very high and the curve is less damaged than `greedy`.
- `draft:random` has a better curve than some greedy heuristics by accident, but
  much lower card value and too many items for the current creature-first battle
  policies.

## First Candidate Drafts

Implemented after the initial report:

- `draft:weighted-balanced`: balanced curve/creature targets, but lower item
  discount so premium removal can beat same-cost medium creatures more often.
- `draft:truecost-balanced`: balanced curve/creature targets, using the full-card
  value proxy keyed by `card_id` so HP swing and card draw are visible at draft
  time. This uses only static public card text, not hidden game state.
- `ground-draft:<name>`: experimental policy spec that pairs a draft-only policy
  with the same `GroundBattlePolicy`, allowing same-battle draft validation.
- `ppo-draft:<name>,<model.zip>`: experimental policy spec that pairs a draft-only
  policy with the same PPO battle model, allowing learned-policy draft validation.

Expanded 1000-draft proxy report:

```bash
COLUMNS=220 uv run locma draft-report draft:random draft:greedy \
  draft:weighted draft:balanced draft:weighted-balanced \
  draft:truecost-balanced draft:max-guard draft:max-attack \
  draft:max-defense --drafts 1000 --seed 1000 --top-cards 12
```

| policy | quality | card value | eff cost delta | avg cost | creatures | items | curve L1 | draw | G/W/L |
|--------|--------:|-----------:|---------------:|---------:|----------:|------:|---------:|-----:|-------|
| `draft:truecost-balanced` | 9.83 | 10.98 | +0.11 | 4.76 | 28.8 | 1.2 | 8.4 | 3.0 | 10.1/2.9/2.1 |
| `draft:weighted-balanced` | 9.78 | 10.90 | +0.10 | 4.74 | 28.5 | 1.5 | 8.2 | 2.0 | 10.3/3.0/2.2 |
| `draft:balanced` | 9.52 | 10.96 | +0.01 | 4.85 | 29.3 | 0.7 | 9.7 | 2.0 | 10.7/3.0/2.2 |
| `draft:weighted` | 9.13 | 12.01 | -0.09 | 5.40 | 27.7 | 2.3 | 18.6 | 2.4 | 9.6/2.9/1.9 |
| `draft:max-guard` | 8.96 | 10.86 | +0.03 | 4.80 | 29.4 | 0.6 | 12.8 | 2.1 | 15.3/2.8/1.7 |
| `draft:max-defense` | 8.82 | 11.39 | -0.15 | 5.19 | 29.4 | 0.6 | 16.2 | 2.5 | 9.6/2.5/1.9 |
| `draft:greedy` | 8.58 | 11.63 | -0.25 | 5.40 | 28.5 | 1.5 | 18.6 | 2.5 | 8.5/2.7/1.8 |
| `draft:max-attack` | 8.56 | 11.32 | -0.22 | 5.24 | 29.4 | 0.6 | 16.9 | 2.8 | 7.5/2.8/1.7 |
| `draft:random` | 6.66 | 8.37 | -0.01 | 3.78 | 21.7 | 8.3 | 11.3 | 3.4 | 6.4/2.1/1.7 |

The proxy says both new variants are improvements. Same-battle validation is
less enthusiastic:

```bash
uv run locma tournament ground-draft:balanced ground-draft:weighted-balanced \
  ground-draft:truecost-balanced ground-draft:max-guard ground-draft:max-attack \
  --games 100 --seed 21000000 --reference ground-draft:balanced --matrix
```

Pair matrix highlights, 200 actual games per pair:

| row vs column | balanced | weighted-balanced | truecost-balanced | max-guard | max-attack |
|---------------|---------:|------------------:|------------------:|----------:|-----------:|
| `balanced` | -- | 0.53 | 0.54 | 0.58 | 0.59 |
| `weighted-balanced` | 0.47 | -- | 0.54 | 0.56 | 0.56 |
| `truecost-balanced` | 0.47 | 0.47 | -- | 0.53 | 0.56 |
| `max-guard` | 0.42 | 0.43 | 0.47 | -- | 0.54 |
| `max-attack` | 0.41 | 0.44 | 0.44 | 0.47 | -- |

Focused 1000-game mirrored head-to-heads:

| matchup | row win rate | 95% CI | p |
|---------|-------------:|--------|--:|
| `balanced` vs `weighted-balanced` | 0.492 | 0.461-0.523 | 0.6353 |
| `balanced` vs `truecost-balanced` | 0.511 | 0.480-0.542 | 0.5067 |
| `weighted-balanced` vs `truecost-balanced` | 0.505 | 0.474-0.536 | 0.7760 |

Read: the proxy-generated candidates are plausible and clearly better than
single-axis `max-guard`/`max-attack` under ground battle, but they are **not
proven upgrades over `balanced`**. The cheap proxy improved, yet game results are
inside noise. This is exactly why `draft-report` should remain a filter, not a
final metric.

### Local Parameter Sweep

I also swept nearby weighted-balanced knobs without adding more permanent policy
classes:

- curve weight: `2, 3, 4, 5, 6`
- item discount: `4, 6, 8, 10, 12, 14`
- fixed creature bonus/target: `2.0` / `24`
- proxy sample: 300 drafts, seed `31000000`

Top proxy rows:

| candidate | quality | card value | eff cost delta | avg cost | creatures | items | curve L1 | draw |
|-----------|--------:|-----------:|---------------:|---------:|----------:|------:|---------:|-----:|
| `param:c4:i4` | 9.807 | 10.887 | +0.112 | 4.721 | 28.45 | 1.55 | 7.95 | 1.88 |
| `param:c3:i4` | 9.801 | 11.119 | +0.091 | 4.840 | 28.61 | 1.39 | 9.39 | 1.81 |
| `param:c5:i4` | 9.777 | 10.674 | +0.124 | 4.620 | 28.24 | 1.76 | 6.81 | 1.85 |
| `param:c4:i6` | 9.743 | 10.846 | +0.099 | 4.717 | 28.75 | 1.25 | 8.01 | 1.85 |
| `param:c3:i6` | 9.741 | 11.075 | +0.077 | 4.836 | 28.84 | 1.16 | 9.41 | 1.76 |

Same-battle validation against `balanced` with `GroundBattlePolicy`:

| matchup | candidate win rate | 95% CI | p | n |
|---------|-------------------:|--------|--:|--:|
| `param:c3:i4` vs `balanced` | 0.495 | not computed | -- | 1000 |
| `weighted-balanced` (`c4:i4`) vs `balanced` | 0.493 | not computed | -- | 1000 |
| `param:c5:i4` vs `balanced` | 0.524 | not computed | -- | 1000 |
| `param:c5:i4` vs `balanced`, rerun | 0.503 | 0.485-0.521 | 0.7287 | 3000 |
| `param:c6:i4` vs `balanced` | 0.515 | 0.494-0.537 | 0.1726 | 2000 |

Conclusion: the proxy optimum is broad and flat. No nearby curve/item setting has
yet produced a robust same-battle upgrade over `balanced`. Keep
`weighted-balanced` and `truecost-balanced` as research probes because they test
distinct hypotheses, but do not promote either as the default draft policy.

### PPO Battle Draft Swaps

Because the original motivation is learned PPO strength, I also checked the same
draft variants with the restored PPO battle model:

```bash
uv run locma play ppo-draft:balanced,runs/ppo-shuffled-pool.zip scripted \
  --games 150 --seed 33000000
```

Summary, 300 actual mirrored games per baseline:

| PPO draft | scripted | max-guard | max-attack | avg hard3 |
|-----------|---------:|----------:|-----------:|----------:|
| `balanced` | 0.603 | 0.537 | 0.560 | 0.567 |
| `weighted-balanced` | 0.543 | 0.483 | 0.527 | 0.518 |
| `truecost-balanced` | 0.523 | 0.490 | 0.507 | 0.507 |

PPO draft-swap head-to-heads, 400 actual mirrored games:

| matchup | row win rate |
|---------|-------------:|
| `balanced` vs `weighted-balanced` | 0.510 |
| `balanced` vs `truecost-balanced` | 0.568 |
| `weighted-balanced` vs `truecost-balanced` | 0.552 |

Read: for PPO, the value-relaxed drafts are worse than `balanced` on hard
baselines and do not beat it head-to-head. PPO appears to prefer the stricter
creature/curve discipline of `balanced`; extra removal/draw that looks good in
the static proxy does not translate through this battle net.

### Parametric Archetype Drafts

Implemented explicit archetype drafts:

- `draft:aggro`: lower curve, attack/Charge/Breakthrough bias, high-cost penalty.
- `draft:midrange`: balanced stat/keyword weights with a 3-4 cost center.
- `draft:defense`: Guard/Ward/defense bias with a slightly heavier curve.

Proxy report:

```bash
COLUMNS=220 uv run locma draft-report draft:balanced draft:aggro \
  draft:midrange draft:defense draft:weighted-balanced \
  draft:truecost-balanced --drafts 1000 --seed 50000000 --top-cards 0
```

| policy | quality | card value | eff cost delta | avg cost | creatures | items | curve L1 | draw | G/W/L |
|--------|--------:|-----------:|---------------:|---------:|----------:|------:|---------:|-----:|-------|
| `draft:truecost-balanced` | 9.86 | 10.97 | +0.12 | 4.75 | 28.8 | 1.2 | 8.2 | 2.9 | 10.1/3.0/2.1 |
| `draft:weighted-balanced` | 9.80 | 10.91 | +0.11 | 4.73 | 28.5 | 1.5 | 8.2 | 2.0 | 10.3/3.0/2.1 |
| `draft:midrange` | 9.76 | 11.08 | +0.16 | 4.75 | 28.6 | 1.4 | 9.9 | 2.4 | 10.3/2.9/2.2 |
| `draft:defense` | 9.65 | 11.15 | +0.07 | 4.87 | 29.0 | 1.0 | 10.5 | 2.4 | 11.4/3.1/2.2 |
| `draft:balanced` | 9.53 | 10.95 | +0.02 | 4.84 | 29.3 | 0.7 | 9.6 | 1.9 | 10.6/3.1/2.1 |
| `draft:aggro` | 9.20 | 9.91 | +0.26 | 4.16 | 28.7 | 1.3 | 6.5 | 2.2 | 8.7/2.9/2.0 |

Same-battle ground tournament:

```bash
COLUMNS=220 uv run locma tournament ground-draft:balanced \
  ground-draft:aggro ground-draft:midrange ground-draft:defense \
  --games 150 --seed 51000000 --reference ground-draft:balanced --matrix
```

| row vs column | balanced | aggro | midrange | defense |
|---------------|---------:|------:|---------:|--------:|
| `balanced` | -- | 0.35 | 0.50 | 0.57 |
| `aggro` | 0.65 | -- | 0.65 | 0.70 |
| `midrange` | 0.50 | 0.35 | -- | 0.57 |
| `defense` | 0.43 | 0.30 | 0.43 | -- |

PPO quick hard-baseline panel, 200 actual mirrored games per cell:

| PPO draft | scripted | max-guard | max-attack | avg hard3 |
|-----------|---------:|----------:|-----------:|----------:|
| `balanced` | 0.625 | 0.525 | 0.595 | 0.582 |
| `aggro` | 0.710 | 0.640 | 0.730 | 0.693 |
| `midrange` | 0.500 | 0.505 | 0.555 | 0.520 |
| `defense` | 0.525 | 0.455 | 0.565 | 0.515 |

Higher-sample confirmation:

```bash
uv run locma play ppo-draft:aggro,runs/ppo-shuffled-pool.zip \
  ppo-draft:balanced,runs/ppo-shuffled-pool.zip --games 1000 --seed 53000000
uv run locma play ground-draft:aggro ground-draft:balanced \
  --games 1000 --seed 55000000
```

| matchup | row win rate | 95% CI | p | n |
|---------|-------------:|--------|--:|--:|
| `ppo-draft:aggro` vs `ppo-draft:balanced` | 0.591 | 0.569-0.612 | 5.772e-16 | 2000 |
| `ground-draft:aggro` vs `ground-draft:balanced` | 0.653 | 0.632-0.674 | 3.768e-43 | 2000 |

Higher-sample PPO hard-baseline panel, 600 actual mirrored games per cell:

| PPO draft | scripted | max-guard | max-attack | avg hard3 |
|-----------|---------:|----------:|-----------:|----------:|
| `balanced` | 0.575 | 0.555 | 0.607 | 0.579 |
| `aggro` | 0.703 | 0.600 | 0.717 | 0.673 |

Read: `aggro` is the first hand-designed draft that clearly improves the actual
validation panels despite a worse static proxy. This is a useful failure mode for
`draft-report`: lower curve and direct damage pressure matter more to these
battle policies than the static card-value score thinks. `midrange` and
`defense` look like research controls, not upgrades. The default `ppo:` pairing
has therefore been promoted from `balanced` to `aggro`; reproduce the previous
pairing with `ppo-draft:balanced,<model>`.

### Existing Draft Strategy Baseline

Full same-battle draft baseline:

```bash
COLUMNS=260 uv run locma tournament ground-draft:random ground-draft:greedy \
  ground-draft:weighted ground-draft:balanced ground-draft:weighted-balanced \
  ground-draft:truecost-balanced ground-draft:aggro ground-draft:midrange \
  ground-draft:defense ground-draft:max-guard ground-draft:max-attack \
  ground-draft:max-defense --games 200 --seed 61000000 \
  --reference ground-draft:balanced --matrix
```

Ratings from the 12-policy round-robin, 400 actual mirrored games per pair:

| draft strategy | OpenSkill mu | Elo-like | p vs `balanced` |
|----------------|-------------:|---------:|----------------:|
| `aggro` | 25.65 | 1639 | 7.055e-09 |
| `weighted-balanced` | 22.30 | 1548 | 0.9601 |
| `balanced` | 22.14 | 1546 | -- |
| `truecost-balanced` | 21.55 | 1531 | 0.1769 |
| `midrange` | 21.50 | 1533 | 0.2501 |
| `max-guard` | 21.48 | 1529 | 0.2112 |
| `defense` | 20.50 | 1503 | 0.03143 |
| `max-attack` | 19.94 | 1485 | 0.01418 |
| `random` | 18.79 | 1449 | 4.9e-06 |
| `max-defense` | 17.37 | 1420 | 4.129e-15 |
| `weighted` | 17.19 | 1411 | 4.129e-15 |
| `greedy` | 16.97 | 1406 | 2.123e-14 |

Key matrix read: `aggro` beats every other draft strategy with the same
`GroundBattlePolicy`, including `balanced` at 0.65, `weighted-balanced` at 0.60,
`truecost-balanced` at 0.61, `midrange` at 0.64, `defense` at 0.69,
`max-guard` at 0.64, and `max-attack` at 0.76. The older `balanced` and
`weighted-balanced` drafts remain roughly tied; static `weighted`/`greedy`
drafts are weak despite picking high printed-stat cards.

PPO draft-pairing baseline against the standard hard3 scripted panel:

```bash
uv run python - <<'PY'
from locma.policies.registry import make_policy
from locma.harness.match import run_match

model = "runs/ppo-shuffled-pool.zip"
drafts = [
    "random", "greedy", "weighted", "balanced", "weighted-balanced",
    "truecost-balanced", "aggro", "midrange", "defense", "max-guard",
    "max-attack", "max-defense",
]
opps = ["scripted", "max-guard", "max-attack"]
for i, draft in enumerate(drafts):
    row = []
    for j, opp in enumerate(opps):
        result = run_match(
            make_policy(f"ppo-draft:{draft},{model}"),
            make_policy(opp),
            games=100,
            seed=62000000 + i * 1000 + j * 100,
        )
        row.append(result.win_rate_a)
    print(draft, row, sum(row) / len(row))
PY
```

Result, 200 actual mirrored games per cell:

| PPO draft | scripted | max-guard | max-attack | avg hard3 |
|-----------|---------:|----------:|-----------:|----------:|
| `aggro` | 0.680 | 0.615 | 0.750 | 0.682 |
| `max-guard` | 0.650 | 0.525 | 0.530 | 0.568 |
| `midrange` | 0.550 | 0.560 | 0.550 | 0.553 |
| `weighted-balanced` | 0.530 | 0.465 | 0.660 | 0.552 |
| `balanced` | 0.595 | 0.525 | 0.520 | 0.547 |
| `truecost-balanced` | 0.565 | 0.470 | 0.575 | 0.537 |
| `defense` | 0.515 | 0.555 | 0.515 | 0.528 |
| `random` | 0.485 | 0.405 | 0.500 | 0.463 |
| `greedy` | 0.405 | 0.435 | 0.440 | 0.427 |
| `max-defense` | 0.450 | 0.410 | 0.410 | 0.423 |
| `max-attack` | 0.375 | 0.415 | 0.470 | 0.420 |
| `weighted` | 0.355 | 0.370 | 0.340 | 0.355 |

Read: both same-battle and PPO-pairing baselines now point at `aggro` as the
best simple draft strategy. The next tier is noisy and opponent-sensitive, but
`max-guard`, `midrange`, `weighted-balanced`, `balanced`, and
`truecost-balanced` are all plausible controls. `weighted`, `greedy`,
`max-attack`, and `max-defense` are mostly useful as negative controls.

DMCTS same-battle draft baseline:

```bash
uv run python - <<'PY'
# Compose DMCTS battle + each draft strategy, then run the same 12-policy
# round robin. Raw output: runs/draft-dmcts4x8-round-robin-100-seed63000000.json
PY
```

Settings: fair deterministic DMCTS, `K=4`, `I=8`, `rollout_turns=3`, 100
mirrored pairs per matchup, seed `63000000`, 200 actual games per pair.

| draft strategy | Elo-like | p vs `balanced` |
|----------------|---------:|----------------:|
| `aggro` | 1536 | 0.8321 |
| `balanced` | 1535 | -- |
| `weighted-balanced` | 1529 | 0.6207 |
| `max-attack` | 1522 | 0.1374 |
| `truecost-balanced` | 1513 | 0.1790 |
| `midrange` | 1509 | 0.5246 |
| `greedy` | 1506 | 0.1036 |
| `weighted` | 1499 | 0.6207 |
| `defense` | 1491 | 0.0008451 |
| `max-guard` | 1484 | 0.05597 |
| `max-defense` | 1481 | 0.005685 |
| `random` | 1396 | 7.087e-06 |

Focused higher-sample checks against `balanced`:

```bash
# Raw output: runs/draft-dmcts4x8-focused-500-seed64000000.json
```

Settings: same DMCTS4x8 battle, 500 mirrored pairs per matchup, seed
`64000000`, 1000 actual games per pair.

| matchup | row win rate | 95% CI | p | n |
|---------|-------------:|--------|--:|--:|
| `aggro` vs `balanced` | 0.542 | 0.511-0.573 | 0.00864 | 1000 |
| `weighted-balanced` vs `balanced` | 0.501 | 0.470-0.532 | 0.9748 | 1000 |
| `max-attack` vs `balanced` | 0.486 | 0.455-0.517 | 0.3932 | 1000 |
| `truecost-balanced` vs `balanced` | 0.471 | 0.440-0.502 | 0.07141 | 1000 |

Read: `aggro` still looks best, but its advantage is much smaller with DMCTS
search than with `GroundBattlePolicy` or the restored PPO battle model. Stronger
search appears to flatten draft differences and makes `balanced` a much more
competitive default. The result still supports `aggro` for learned PPO pairing,
but it weakens the claim that `aggro` is universally better deck construction.

### Impact-Guided Draft Probe

I also added a reproducible PPO-impact draft probe:

```bash
COLUMNS=220 uv run locma impact-draft-sweep ppo:runs/ppo-shuffled-pool.zip \
  --fit-games 1200 --fit-seed 56000000 --fit-alpha 20 \
  --eval-games 200 --eval-seed 57000000 --reference balanced \
  --spec 10:3:8 --spec 15:3:8 --spec 20:3:8 \
  --spec 15:4:8 --spec 20:4:8 --spec 20:4:12
```

It fits card-impact coefficients from `ppo:runs/ppo-shuffled-pool.zip`, then
drafts by `scale * empirical_card_impact + curve_weight * curve_need -
item_discount`.

Reproducible sweep, 400 actual mirrored games per candidate:

| candidate | win rate vs balanced | 95% CI | p |
|-----------|---------------------:|--------|--:|
| `impact-s20-c3-i8` | 0.613 | 0.564-0.659 | 7.905e-06 |
| `impact-s20-c4-i8` | 0.580 | 0.531-0.627 | 0.0016 |
| `impact-s10-c3-i8` | 0.560 | 0.511-0.608 | 0.0187 |
| `impact-s15-c4-i8` | 0.557 | 0.509-0.605 | 0.0243 |
| `impact-s15-c3-i8` | 0.547 | 0.499-0.596 | 0.0642 |
| `impact-s20-c4-i12` | 0.532 | 0.484-0.581 | 0.2112 |

Earlier ad hoc held-out PPO head-to-heads vs `balanced`, 400 actual mirrored
games each:

| candidate | win rate vs balanced | 95% CI | p |
|-----------|---------------------:|--------|--:|
| `impact-s10-c3-i8` | 0.547 | 0.499-0.596 | 0.0642 |
| `impact-s15-c3-i8` | 0.568 | 0.519-0.615 | 0.0080 |
| `impact-s20-c3-i8` | 0.578 | 0.529-0.625 | 0.0022 |
| `impact-s15-c4-i8` | 0.575 | 0.526-0.623 | 0.0031 |
| `impact-s20-c4-i8` | 0.588 | 0.539-0.635 | 0.0005 |
| `impact-s20-c4-i12` | 0.585 | 0.536-0.632 | 0.0008 |

Quick hard-baseline panel, 200 actual mirrored games per cell:

| PPO draft | scripted | max-guard | max-attack | avg hard3 |
|-----------|---------:|----------:|-----------:|----------:|
| `balanced` | 0.550 | 0.490 | 0.530 | 0.523 |
| `impact-s20-c4-i8` | 0.685 | 0.600 | 0.690 | 0.658 |

Read: empirical PPO card-impact weights are a stronger path than static
true-cost heuristics. The command is now reproducible, but not yet promoted to a
normal draft policy because each run depends on fitted weights. Next step is
either a saved weight-table artifact or a policy spec that points at such a table.

### Saved Impact Artifact

The artifact path now exists:

```bash
uv run locma card-impact --games 1200 --seed 58000000 \
  --battle ppo:runs/ppo-shuffled-pool.zip --alpha 20 \
  --top-cards 8 --out runs/ppo-impact-1200-seed58000000.json
```

And can be evaluated as a normal policy:

```bash
uv run locma play \
  ppo-impact-draft:runs/ppo-impact-1200-seed58000000.json,runs/ppo-shuffled-pool.zip,20,3,8 \
  ppo-draft:balanced,runs/ppo-shuffled-pool.zip \
  --games 500 --seed 59000000
```

Result: impact artifact draft beats the old balanced pairing:

| matchup | row win rate | 95% CI | p | n |
|---------|-------------:|--------|--:|--:|
| `ppo-impact-draft:...,20,3,8` vs `ppo-draft:balanced` | 0.583 | 0.552-0.613 | 1.702e-07 | 1000 |

Against the new `ppo:` default (`aggro`), the same artifact is even:

```bash
uv run locma play \
  ppo-impact-draft:runs/ppo-impact-1200-seed58000000.json,runs/ppo-shuffled-pool.zip,20,3,8 \
  ppo:runs/ppo-shuffled-pool.zip \
  --games 500 --seed 60000000
```

| matchup | row win rate | 95% CI | p | n |
|---------|-------------:|--------|--:|--:|
| `ppo-impact-draft:...,20,3,8` vs `ppo` (`aggro`) | 0.508 | 0.477-0.539 | 0.6353 | 1000 |

Read: saved impact weights are now reproducible and useful for research, but they
do not currently beat the much simpler `aggro` default. Keep
`ppo-impact-draft` hidden/experimental.

## True Mana Cost Side Quest

The first pass estimates effective cost by fitting printed mana cost from the
full-card heuristic value. It is useful for surfacing outliers, not for claiming
final card balance.

Top underpriced by this proxy:

| id | card | type | cost | effective cost | delta |
|---:|------|------|-----:|---------------:|------:|
| 151 | Decimate | red_item | 5 | 10.40 | +5.40 |
| 142 | Staff of Suppression | red_item | 0 | 4.92 | +4.92 |
| 148 | Helm Crusher | red_item | 2 | 5.77 | +3.77 |
| 149 | Rootkin Ritual | red_item | 3 | 5.85 | +2.85 |
| 91 | Flumpy | creature | 0 | 1.93 | +1.93 |
| 83 | Restless Owl | creature | 0 | 1.60 | +1.60 |
| 118 | Royal Helm | green_item | 0 | 1.51 | +1.51 |
| 63 | Rootkin Drone | creature | 2 | 3.49 | +1.49 |
| 38 | Imp | creature | 1 | 2.40 | +1.40 |
| 93 | Spinekid | creature | 1 | 2.35 | +1.35 |
| 73 | Bog Bounder | creature | 4 | 5.30 | +1.30 |
| 47 | Gnipper | creature | 2 | 3.24 | +1.24 |

Top overpriced by this proxy:

| id | card | type | cost | effective cost | delta |
|---:|------|------|-----:|---------------:|------:|
| 90 | Eldritch Swooper | creature | 8 | 4.97 | -3.03 |
| 152 | Mighty Throwing Axe | red_item | 7 | 4.12 | -2.88 |
| 81 | Flying Corpse Guzzler | creature | 9 | 6.14 | -2.86 |
| 46 | Soul Devourer | creature | 9 | 6.61 | -2.39 |
| 133 | Pie of Power | green_item | 5 | 2.65 | -2.35 |
| 139 | Grow Stingers | green_item | 4 | 1.72 | -2.28 |
| 115 | Rootkin Warchief | creature | 8 | 6.02 | -1.98 |
| 132 | High Protein | green_item | 5 | 3.11 | -1.89 |
| 60 | Gritsuck Troll | creature | 7 | 5.30 | -1.70 |
| 53 | Possessed Abomination | creature | 4 | 2.35 | -1.65 |
| 131 | Heavy Gauntlet | green_item | 4 | 2.35 | -1.65 |
| 35 | Snail-eyed Hulker | creature | 6 | 4.46 | -1.54 |

Known caveats:

- Removal and ability stripping are context-dependent. The proxy caps destroy
  sentinels and assigns a fixed ability-strip bonus, so it will overrate dead
  removal in creature-light matchups and underrate premium removal when boards are
  dense.
- Creature keywords are not independent. Ward plus Guard, Ward plus Lethal, and
  Charge plus high attack should probably be interaction terms.
- Draw is valued statically. A battle policy that dumps hand quickly should value
  draw more than one that floats cards behind the hand cap.
- Green items depend on friendly-board reliability. Current PPO-like policies
  have historically played creature decks much better than spell-heavy decks, so
  the value model should remain battle-policy-specific.

## Empirical Card Impact

Static true-cost estimates are only card-text priors. I added a gameplay-derived
complement:

```bash
uv run locma card-impact --games 2000 --seed 40000000 \
  --battle ground --alpha 20 --top-cards 15
```

Method:

- Run random-draft games with both seats using the same fixed battle policy.
- Use independent deterministic draft RNG streams for the two seats.
- Feature row = player0 deck card counts minus player1 deck card counts.
- Target = `+1` if player0 wins, `-1` if player1 wins.
- Fit a ridge regression over the 160 card ids.

This is not causal truth. It is an empirical card contribution estimate under one
draft distribution and one battle policy.

### Ground Battle, 2000 Games

Top positive coefficients:

| id | card | type | cost | impact | eff cost delta |
|---:|------|------|-----:|-------:|---------------:|
| 10 | Carnivorous Bush | creature | 3 | +0.1690 | +0.18 |
| 106 | Far-reaching Nightmare | creature | 5 | +0.1483 | -1.75 |
| 96 | Prairie Protector | creature | 2 | +0.1385 | +1.29 |
| 7 | Rootkin Sapling | creature | 2 | +0.1342 | +1.30 |
| 105 | King Shellcrab | creature | 5 | +0.1328 | -1.69 |
| 5 | Grime Gnasher | creature | 2 | +0.1272 | +1.32 |
| 41 | Blizzard Demon | creature | 3 | +0.1208 | +0.35 |
| 26 | Razor Crab | creature | 2 | +0.1199 | +1.35 |

Top negative coefficients:

| id | card | type | cost | impact | eff cost delta |
|---:|------|------|-----:|-------:|---------------:|
| 129 | Enchanted Leather | green_item | 4 | -0.1611 | +0.32 |
| 110 | Gargoyle | creature | 5 | -0.1235 | -0.81 |
| 153 | Healing Potion | blue_item | 2 | -0.1195 | +2.17 |
| 145 | Cursed Sword | red_item | 3 | -0.1185 | +1.17 |
| 119 | Serrated Shield | green_item | 1 | -0.1120 | +3.15 |
| 150 | Throwing Axe | red_item | 2 | -0.1111 | +2.14 |
| 60 | Gritsuck Troll | creature | 7 | -0.1041 | -2.88 |
| 74 | Crusher | creature | 5 | -0.0964 | -0.91 |

### Greedy Battle Cross-Check, 1000 Games

```bash
uv run locma card-impact --games 1000 --seed 41000000 \
  --battle greedy --alpha 20 --top-cards 12
```

Greedy battle shifts value toward larger threats:

| id | card | type | cost | impact | eff cost delta |
|---:|------|------|-----:|-------:|---------------:|
| 116 | Emperor Nightmare | creature | 12 | +0.2020 | -6.16 |
| 19 | Foulbeast | creature | 5 | +0.1872 | +0.69 |
| 80 | Corpse Guzzler | creature | 8 | +0.1808 | -2.38 |
| 115 | Rootkin Warchief | creature | 8 | +0.1799 | -2.39 |
| 68 | Giant Squid | creature | 6 | +0.1736 | -0.45 |
| 46 | Soul Devourer | creature | 9 | +0.1686 | -3.50 |

And it heavily penalizes many items:

| id | card | type | cost | impact | eff cost delta |
|---:|------|------|-----:|-------:|---------------:|
| 134 | Light The Way | green_item | 4 | -0.2733 | -3.05 |
| 130 | Helm of Remedy | green_item | 4 | -0.2374 | -2.68 |
| 124 | Blood Grapes | green_item | 3 | -0.2022 | -1.32 |
| 137 | Ward | green_item | 2 | -0.1967 | -0.26 |
| 117 | Protein | green_item | 1 | -0.1947 | +0.76 |
| 150 | Throwing Axe | red_item | 2 | -0.1936 | -0.23 |

Read: “true card cost” is battle-policy-specific. The static proxy identifies
textually underpriced effects, but empirical impact says whether the current
battle policy can actually convert those effects into wins. For the current
simple battle policies and PPO, creature reliability dominates many attractive
spell/item effects.

### PPO Battle, 1000 Games

`card-impact` also accepts full policy specs and reuses their battle half. This
lets the true-cost side quest target the restored PPO directly:

```bash
uv run locma card-impact --games 1000 --seed 43000000 \
  --battle ppo:runs/ppo-shuffled-pool.zip --alpha 20 --top-cards 15
```

Top positive coefficients:

| id | card | type | cost | impact | eff cost delta |
|---:|------|------|-----:|-------:|---------------:|
| 51 | Elite Bilespitter | creature | 4 | +0.1780 | -0.02 |
| 62 | Mutant Troll | creature | 12 | +0.1593 | -8.04 |
| 64 | Coppershell Tortoise | creature | 2 | +0.1592 | +1.96 |
| 82 | Slithering Nightmare | creature | 7 | +0.1563 | -3.05 |
| 75 | Titan Prowler | creature | 5 | +0.1444 | -1.06 |
| 115 | Rootkin Warchief | creature | 8 | +0.1378 | -4.07 |
| 70 | Murglord | creature | 4 | +0.1273 | -0.08 |
| 39 | Voracious Imp | creature | 1 | +0.1233 | +2.91 |
| 5 | Grime Gnasher | creature | 2 | +0.1204 | +1.91 |
| 53 | Possessed Abomination | creature | 4 | +0.1179 | -0.09 |

Top negative coefficients:

| id | card | type | cost | impact | eff cost delta |
|---:|------|------|-----:|-------:|---------------:|
| 57 | Dream-Eater | creature | 4 | -0.1772 | -0.46 |
| 56 | Giant Louse | creature | 4 | -0.1401 | -0.41 |
| 120 | Venomfruit | green_item | 2 | -0.1324 | +1.60 |
| 151 | Decimate | red_item | 5 | -0.1307 | -1.40 |
| 155 | Scroll of Firebolt | blue_item | 3 | -0.1283 | +0.60 |
| 124 | Blood Grapes | green_item | 3 | -0.1263 | +0.61 |
| 55 | Hermit Slime | creature | 2 | -0.1229 | +1.61 |
| 157 | Life Sap Drop | blue_item | 3 | -0.1224 | +0.61 |
| 154 | Poison | blue_item | 2 | -0.1165 | +1.62 |
| 152 | Mighty Throwing Axe | red_item | 7 | -0.1090 | -3.37 |

Read: PPO's empirical card impact reinforces the draft-swap result. Even premium
static effects like Decimate are negative under random-draft PPO play, while the
positive list is almost entirely creatures. This explains why relaxing
`balanced` toward more removal/draw raised proxy quality but reduced PPO
baseline results.

## Better Draft Policy Directions

The next draft experiments should be staged from cheap to expensive:

1. **Card-cost-surplus draft.** Pick by `card_value + alpha * effective_cost_delta`
   with curve/creature constraints. This directly uses the true-cost side quest.
2. **Archetype-aware drafts.** Maintain separate target curves and weights for
   Guard-control, attack-tempo, and removal-light creature decks. Evaluate each
   against the same battle policy before mixing.
3. **Offline draft policy search.** Treat draft weights as a small vector and run
   random search/CMA-style sweeps. Cheap filter: `draft-report`; expensive filter:
   same-battle tournament against hard baselines and `dmcts`.
4. **Learned draft evaluator.** Generate many draft pools, draft with candidate
   policies, play fixed-battle matches, and train a model to predict deck win
   rate from deck features. This would turn deck quality from a hand proxy into a
   supervised evaluator.

## Tournament Validation Plan

Proxy score is only a candidate filter. A draft policy should graduate only after
same-battle validation:

- Compose each draft with the same battle policy (`GroundBattlePolicy`,
  `GreedyBattlePolicy`, and PPO battle if the `[ml]` extra/model are available).
- Run mirrored tournaments with identical seeds.
- Track both full matrix and average hard-baseline score
  (`scripted`, `max-guard`, `max-attack`).
- Keep `dmcts` as the fair search reference once a draft beats the no-search
  baselines.

Suggested first validation for future draft candidates:

```bash
uv run locma draft-report draft:balanced draft:weighted-balanced draft:NEW \
  --drafts 1000 --seed 1000 --top-cards 20
```

Then run same-battle tournaments with `ground-draft:NEW`. For PPO-specific deck
work, run the same matrix through `ppo-draft:<draft>,<model>`. Otherwise draft
strength will stay entangled with battle-policy differences.
