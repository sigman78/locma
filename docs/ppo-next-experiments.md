# PPO next experiments

_Date: 2026-06-27_

See also: `docs/ppo-method-scorecard.md` for the current hypothesis, method
ranking, and repeatable scorecard command.

## Current read

The current checkout already has the important PPO fixes that the early weak
policy lacked:

- fixed semantic 155-action space and legal mask;
- 308-d observation with card type, all abilities, and readiness;
- `ppo:` paired with the `balanced` draft;
- both-seat PPO training by default;
- training env passes the forward-model state to search opponents.

The remaining gap is therefore probably not the old action encoding bug. The
strongest documented PPO beats the non-search baselines but loses badly to fair
search (`dmcts`). Prior work has already ruled out the obvious knobs: more PPO
steps, opponent diversity, entropy, observation normalization, larger MLP,
reward shaping, self-play of the raw net, and distillation of search into the
same reactive policy.

Working hypothesis: the residual gap is the reactive-policy ceiling. A network
that maps visible observation directly to one action can learn strong tempo
habits, but it does not get the tactical lookahead that `dmcts` gets from search.

Current research target: improve the **no-search learned policy** family
(`ppo:` / `ppo-tactical:` style inference). Search wrappers such as `puct-ppo`
and `dpuct-ppo` are useful diagnostics for whether PPO contains usable priors,
but they do not satisfy the target unless their lessons transfer back into a
single forward pass policy.

## 2026-06-27 implementation pass

Added experiment hooks without changing the deployed base model layout:

- `--obs-mode tactical` for PPO training, plus `ppo-tactical:path` for eval;
- `--reward-mode sparse|health|board`, defaulting to the old sparse reward;
- `--init-model path` warm-start for continuing PPO from a restored or distilled
  artifact;
- `dmcts:K,I,seed,turns,1` deterministic mode for behavior-cloning data;
- `locma action-stats POLICY --opponent OPP` for quick tactical histograms.
- `puct-ppo:iterations,model_path,c_puct,seed,turns,obs` for cheating
  perfect-foresight PUCT search with PPO policy-head priors and the existing
  AZ-lite board/health leaf;
- `dpuct-ppo:K,I,model_path,c_puct,seed,turns,obs` for a fair determinized
  wrapper around PPO-prior PUCT.
- `train-zoo --n-envs N` so longer no-search PPO curriculum runs can use the
  multicore machine.
- `--learning-rate` for no-search PPO continuation sweeps.

Validation:

```bash
uv run pytest tests/test_encode.py tests/test_env.py tests/test_registry.py \
  tests/test_ppo.py tests/test_training_zoo.py tests/test_mcts.py tests/test_action_stats.py
```

Result: 41 passed.

### Tactical observation smoke

`runs/ppo-tactical-mixed-100k.zip` was trained with:

```bash
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-tactical-mixed-100k.zip --seed 0 --obs-mode tactical --n-envs 1
```

Small held-out eval (`--games 30`, 60 mirrored games):

| policy | random | scripted | greedy | max-guard | max-attack |
|--------|--------|----------|--------|-----------|------------|
| `ppo-tactical` 100k | 1.000 | 0.500 | 0.567 | 0.450 | 0.517 |

This is a proof-of-life, not a challenger to the 800k restored model. The full
800k tactical zoo is slower than the base run; run it as a long job or with
checkpoints.

### Health reward warm-start

Continued the restored strong model for +100k on mixed opponents:

```bash
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-shuffled-health-100k.zip --seed 1 \
  --reward-mode health --init-model runs/ppo-shuffled-pool.zip --n-envs 1
```

Same-seed eval (`--games 100`, 200 mirrored games):

| model | scripted | greedy | max-guard | max-attack | avg hard3 |
|-------|----------|--------|-----------|------------|-----------|
| restored `ppo-shuffled-pool` | 0.580 | 0.650 | 0.540 | 0.555 | 0.558 |
| +100k health reward | 0.515 | 0.715 | 0.555 | 0.560 | 0.543 |

It traded `scripted` strength for `greedy` strength and netted slightly down on
the hard trio. Treat this reward variant as negative unless a longer paired run
shows otherwise.

Small `dmcts` check on same seeds (`--games 20`, 40 mirrored games):

| model | vs `dmcts` |
|-------|------------|
| restored `ppo-shuffled-pool` | 0.375 |
| +100k health reward | 0.400 |

The interval is too wide to call this a gain, and hard-baseline generalization
got worse.

### Action stats snapshot

Against `max-guard`, PPO and `dmcts` are not wildly different in coarse action
mix. One recurring signal is low item use by PPO.

| policy | pass | summon | use | attack | face atk | unit atk |
|--------|------|--------|-----|--------|----------|----------|
| restored PPO | 0.260 | 0.302 | 0.008 | 0.430 | 0.129 | 0.301 |
| health PPO | 0.263 | 0.304 | 0.009 | 0.424 | 0.150 | 0.274 |
| `dmcts` | 0.237 | 0.272 | 0.015 | 0.475 | 0.150 | 0.325 |

Use this diagnostic for future candidate models before spending long `dmcts`
eval time.

### Deterministic DMCTS practicum smoke

Verified the behavior-cloning data path with deterministic `dmcts`:

```bash
uv run locma record-practicum --teacher dmcts:2,3,0,3,1 \
  --opponents greedy --games 2 --out runs/dmcts-det-smoke.npz --seed 1000000
uv run locma distill --data runs/dmcts-det-smoke.npz \
  --out runs/dmcts-det-bc-smoke.zip --epochs 1 --batch 64 --seed 0
```

This recorded 100 examples with zero dropped and saved a BC model. The dataset is
too small to evaluate; the point is that deterministic teacher practicum +
distill + optional PPO warm-start is now wired.

### PPO-prior PUCT smoke

Added `puct-ppo`, which keeps AZ-lite's PUCT search and heuristic value leaf but
uses a loaded MaskablePPO policy distribution as the prior over legal semantic
action slots. This is **perfect-foresight cheating** like `azlite`: the PPO prior
is computed from a sanitized visible `BattleView`, but the PUCT tree clones and
advances the real `GameState`, including hidden hand/deck/order. The restored
baseline artifact remains the ordinary `ppo:` model; this is a search wrapper
around it.

Added `dpuct-ppo` as the fair counterpart. It samples plausible hidden worlds
using the same determinization discipline as `dmcts`: opponent hidden hand/deck
are resampled and own future deck order is reshuffled before each inner PUCT
search.

Smoke commands:

```bash
uv run locma play puct-ppo:4,runs/ppo-shuffled-pool.zip max-guard \
  --games 2 --seed 1000000
uv run locma play puct-ppo:25,runs/ppo-shuffled-pool.zip max-guard \
  --games 20 --seed 1000000
```

Small held-out smoke (`--games 20`, 40 mirrored games):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 |
|--------|----------|--------|-----------|------------|-----------|
| `puct-ppo:25` | 0.750 | 0.825 | 0.675 | 0.700 | 0.733 |

Same-seed `max-guard` controls at 40 mirrored games:

| policy | win rate |
|--------|----------|
| restored `ppo-shuffled-pool` | 0.575 |
| `azlite:25` | 0.575 |
| `puct-ppo:25` | 0.675 |

This is promising but far too small to declare a strength gain. Tiny
`dmcts:4,8,0,3` was not discriminative: both restored PPO and `puct-ppo:25`
scored 0.900 over 20 mirrored games.

`action-stats` against `max-guard` (`--games 20`) stayed close to the restored
PPO distribution:

| policy | pass | summon | use | attack | face atk | unit atk |
|--------|------|--------|-----|--------|----------|----------|
| `puct-ppo:25` | 0.259 | 0.298 | 0.007 | 0.437 | 0.148 | 0.289 |

Next validation should use the same 300-game held-out matrix recommended below,
including a stronger `dmcts` sample. If this holds up, PPO already has useful
policy priors even though the raw reactive policy is capped.

Fairness/performance smoke (`--games 80`, 160 mirrored games, seed `2000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | seconds/game/process |
|--------|----------|--------|-----------|------------|-----------|----------------------|
| restored `ppo-shuffled-pool` | 0.619 | 0.681 | 0.550 | 0.581 | 0.604 | 0.038 |
| cheating `puct-ppo:25` | 0.731 | 0.744 | 0.675 | 0.700 | 0.706 | 0.140 |
| fair `dpuct-ppo:5,5` | 0.688 | 0.719 | 0.575 | 0.594 | 0.629 | 0.160 |

Read: PPO priors help search, but much of the strong `puct-ppo:25` gain was
perfect-foresight leakage. The fair determinized variant is still slightly above
raw PPO on this small sample, but weaker and slower than the cheating wrapper.

Held-out search comparison (`--games 300`, 600 mirrored games, seed `1000000`):

| policy | vs `dmcts` | vs `azlite:25` |
|--------|------------|----------------|
| restored `ppo-shuffled-pool` | 0.288 | 0.383 |
| fair `dpuct-ppo:5,5` | 0.310 | 0.422 |
| cheating `puct-ppo:25` | 0.357 | 0.495 |

Controls:

| matchup | row win rate |
|---------|--------------|
| `azlite:25` vs `dmcts` | 0.498 |
| `puct-ppo:25` vs `azlite:25` | 0.495 |

Read: fair PPO-prior search moves the restored PPO only modestly against
`dmcts` (`0.288 -> 0.310`) and more visibly against cheating `azlite:25`
(`0.383 -> 0.422`). Cheating `puct-ppo:25` is roughly `azlite:25` strength at
the same 25-iteration budget, but that result should stay out of fair-policy
claims because both clone the real hidden state.

Determinization/depth grid (`--games 80`, 160 mirrored games, seed `3000000`):

| fair config | greedy | max-guard | max-attack | avg hard3 | `dmcts:4,8,0,3` | seconds/game/process |
|-------------|--------|-----------|------------|-----------|-----------------|----------------------|
| `dpuct-ppo:1,25` | 0.731 | 0.700 | 0.769 | 0.733 | 0.963 | 0.156 |
| `dpuct-ppo:3,8` | 0.738 | 0.656 | 0.719 | 0.704 | 0.906 | 0.163 |
| `dpuct-ppo:5,5` | 0.731 | 0.681 | 0.738 | 0.717 | 0.912 | 0.170 |
| `dpuct-ppo:8,3` | 0.713 | 0.606 | 0.719 | 0.679 | 0.894 | 0.178 |
| `dpuct-ppo:5,10` | 0.731 | 0.662 | 0.744 | 0.712 | 0.938 | 0.273 |

Read: at this budget, one sampled world with deeper PUCT (`1x25`) beat many
shallow determinizations on the scripted hard baselines. This is still fair
with respect to hidden state because the one world is sampled, not the real
hidden world, but it is high-variance by design. Validate `dpuct-ppo:1,25`
against default `dmcts`/`azlite:25` before treating it as the fair search
candidate.

### No-search PPO scout: tactical zoo and board reward

Ran two ordinary learned-policy candidates; neither uses search at inference:

```bash
uv run locma train-zoo --out runs/ppo-tactical-zoo-200k.zip --seed 2 \
  --obs-mode tactical --steps-per-opponent 50000 --n-envs 8
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-shuffled-board-100k.zip --seed 2 \
  --reward-mode board --init-model runs/ppo-shuffled-pool.zip --n-envs 8
```

Eval (`--games 100`, 200 mirrored games for scripted baselines;
`--games 50`, 100 mirrored games for default `dmcts`; seed `4000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | `dmcts` |
|--------|----------|--------|-----------|------------|-----------|---------|
| restored `ppo-shuffled-pool` | 0.600 | 0.650 | 0.580 | 0.600 | 0.610 | 0.250 |
| `ppo-shuffled-board-100k` | 0.560 | 0.675 | 0.530 | 0.585 | 0.597 | 0.280 |
| `ppo-tactical-zoo-200k` | 0.510 | 0.655 | 0.450 | 0.550 | 0.552 | 0.180 |

Read: board reward continuation is flat/slightly negative on hard baselines,
with a tiny `dmcts` bump too noisy to trust. The tactical 200k zoo scout is not
competitive; tactical obs may still need a full run, but this scout does not
justify treating tactical features as the main path by itself.

Action stats vs `max-guard` (`--games 80`, seed `4000000`):

| policy | pass | summon | use | attack | face atk | unit atk | lethal avail | lethal take | guard present | guard attack |
|--------|------|--------|-----|--------|----------|----------|--------------|-------------|---------------|--------------|
| restored PPO | 0.270 | 0.306 | 0.009 | 0.415 | 0.143 | 0.272 | 0.035 | 0.889 | 0.479 | 0.211 |
| board +100k | 0.271 | 0.305 | 0.008 | 0.416 | 0.142 | 0.274 | 0.034 | 0.850 | 0.484 | 0.211 |
| tactical zoo 200k | 0.270 | 0.314 | 0.008 | 0.408 | 0.122 | 0.287 | 0.034 | 0.629 | 0.496 | 0.211 |
| `dmcts` | 0.279 | 0.271 | 0.019 | 0.431 | 0.167 | 0.264 | 0.077 | 0.432 | 0.433 | 0.191 |

Read: the no-search variants still under-use items relative to `dmcts`.
The tactical scout did not fix that and appears to hurt lethal-taking behavior
on this sample. This favors architecture/action-representation work over more
scalar reward shaping.

### Reward-alignment sweep

Historical scout: exposed `--reward-scale` and tried `reward-mode=tempo`, a
potential-based scalar:
health lead + board power + hand count + ready attack + guard defense. Also
exposed `--learning-rate` so warm-start PPO continuation can be made less
aggressive.

Code status: `tempo` was pruned after this negative result; main keeps only the
smaller `sparse|health|board` reward scaffold.

Commands:

```bash
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-shuffled-tempo-100k.zip --seed 5 \
  --reward-mode tempo --reward-scale 0.05 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-shuffled-board-s002-100k.zip --seed 6 \
  --reward-mode board --reward-scale 0.02 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-shuffled-sparse-lr1e4-100k.zip --seed 7 \
  --reward-mode sparse --learning-rate 0.0001 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
```

Eval (`--games 100` for scripted baselines, `--games 50` for default `dmcts`,
seed `6000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | `dmcts` |
|--------|----------|--------|-----------|------------|-----------|---------|
| restored `ppo-shuffled-pool` | 0.645 | 0.670 | 0.600 | 0.635 | 0.635 | 0.310 |
| board scale 0.05 | 0.605 | 0.740 | 0.575 | 0.570 | 0.628 | 0.300 |
| board scale 0.02 | 0.610 | 0.665 | 0.575 | 0.595 | 0.612 | 0.320 |
| tempo scale 0.05 | 0.580 | 0.670 | 0.555 | 0.585 | 0.603 | 0.220 |
| sparse LR `1e-4` | 0.645 | 0.695 | 0.590 | 0.610 | 0.632 | 0.340 |

Read: scalar dense rewards did not improve the no-search policy. The best signal
was not reward alignment; it was gentler PPO continuation (`learning_rate=1e-4`),
which preserved hard-baseline strength and had the best small `dmcts` probe.
Next PPO-like tuning should vary LR/entropy/update size around sparse reward
rather than inventing more scalar potentials.

Follow-up sparse tuning:

```bash
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-shuffled-sparse-lr1e4-ent005-100k.zip --seed 8 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.005 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
```

Eval (`--games 100` for scripted baselines, `--games 50` for default `dmcts`,
seed `7000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | `dmcts` |
|--------|----------|--------|-----------|------------|-----------|---------|
| restored `ppo-shuffled-pool` | 0.525 | 0.660 | 0.565 | 0.515 | 0.580 | 0.240 |
| sparse LR `1e-4` | 0.545 | 0.705 | 0.505 | 0.570 | 0.593 | 0.310 |
| sparse LR `1e-4`, entropy `0.005` | 0.550 | 0.715 | 0.530 | 0.580 | 0.608 | 0.220 |

Read: lower entropy helped hard3 in this seed window, but gave back the small
`dmcts` gain from the higher-entropy low-LR continuation. This is a plausible
PPO-like tuning axis, but it needs a larger held-out evaluation before calling it
an improvement.

### Sparse PPO continuation sweep

Wide 100k sweep from the restored PPO, all no-search at inference:

```bash
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-sweep-sparse-lr5e5-ent005-100k.zip --seed 11 \
  --reward-mode sparse --learning-rate 0.00005 --ent-coef 0.005 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-sweep-sparse-lr2e4-ent005-100k.zip --seed 12 \
  --reward-mode sparse --learning-rate 0.0002 --ent-coef 0.005 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-sweep-sparse-lr1e4-ent0-100k.zip --seed 13 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.0 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent mixed --steps 100000 \
  --out runs/ppo-sweep-sparse-lr1e4-ent01-100k.zip --seed 14 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.01 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
```

Eval (`--games 150` for scripted baselines, `--games 60` for default `dmcts`,
seed `8000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | `dmcts` |
|--------|----------|--------|-----------|------------|-----------|---------|
| restored `ppo-shuffled-pool` | 0.613 | 0.717 | 0.563 | 0.603 | 0.628 | 0.283 |
| prev LR `1e-4`, entropy `0.005` | 0.607 | 0.683 | 0.570 | 0.583 | 0.612 | 0.267 |
| LR `5e-5`, entropy `0.005` | 0.613 | 0.710 | 0.593 | 0.580 | 0.628 | 0.308 |
| LR `2e-4`, entropy `0.005` | 0.620 | 0.720 | 0.577 | 0.573 | 0.623 | 0.350 |
| LR `1e-4`, entropy `0.0` | 0.587 | 0.693 | 0.580 | 0.583 | 0.619 | 0.325 |
| LR `1e-4`, entropy `0.01` | 0.620 | 0.733 | 0.587 | 0.587 | 0.636 | 0.258 |

Read: no 100k cell dominated both hard3 and `dmcts`. LR `1e-4`, entropy `0.01`
had the best hard3; LR `2e-4`, entropy `0.005` had the best `dmcts` probe.

Deepened those two to 300k:

```bash
uv run locma train --opponent mixed --steps 300000 \
  --out runs/ppo-deep-sparse-lr2e4-ent005-300k.zip --seed 21 \
  --reward-mode sparse --learning-rate 0.0002 --ent-coef 0.005 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent mixed --steps 300000 \
  --out runs/ppo-deep-sparse-lr1e4-ent01-300k.zip --seed 22 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.01 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
```

Larger eval (`--games 300` for scripted baselines, `--games 100` for default
`dmcts`, seed `9000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | `dmcts` |
|--------|----------|--------|-----------|------------|-----------|---------|
| restored `ppo-shuffled-pool` | 0.578 | 0.655 | 0.540 | 0.588 | 0.594 | 0.305 |
| LR `2e-4`, entropy `0.005`, 100k | 0.605 | 0.693 | 0.547 | 0.605 | 0.615 | 0.320 |
| LR `1e-4`, entropy `0.01`, 100k | 0.567 | 0.690 | 0.555 | 0.623 | 0.623 | 0.260 |
| LR `2e-4`, entropy `0.005`, 300k | 0.605 | 0.695 | 0.573 | 0.605 | 0.624 | 0.305 |
| LR `1e-4`, entropy `0.01`, 300k | 0.630 | 0.697 | 0.568 | 0.618 | 0.628 | 0.260 |

Direct head-to-head vs restored PPO (`--games 300`, seed `9500000`):

| candidate | win rate vs restored |
|-----------|----------------------|
| LR `2e-4`, entropy `0.005`, 100k | 0.537 |
| LR `2e-4`, entropy `0.005`, 300k | 0.517 |
| LR `1e-4`, entropy `0.01`, 300k | 0.540 |

Read: PPO tuning can produce a modest no-search improvement over restored PPO,
especially on hard scripted baselines and direct head-to-head. It still does not
close the fair-search gap: default `dmcts` remains around 0.68-0.74 against the
tuned models on these probes. Best current no-search candidate depends on the
objective:

- hard-baseline/head-to-head: `runs/ppo-deep-sparse-lr1e4-ent01-300k.zip`;
- preserving `dmcts` probe: `runs/ppo-sweep-sparse-lr2e4-ent005-100k.zip`.

Action stats vs `max-guard` (`--games 80`, seed `9000000`):

| policy | pass | summon | use | attack | face atk | unit atk | lethal avail | lethal take | guard present | guard attack |
|--------|------|--------|-----|--------|----------|----------|--------------|-------------|---------------|--------------|
| restored PPO | 0.261 | 0.305 | 0.007 | 0.427 | 0.135 | 0.292 | 0.030 | 0.879 | 0.503 | 0.224 |
| LR `2e-4`, entropy `0.005`, 100k | 0.263 | 0.303 | 0.007 | 0.427 | 0.150 | 0.277 | 0.026 | 0.948 | 0.512 | 0.230 |
| LR `1e-4`, entropy `0.01`, 300k | 0.262 | 0.303 | 0.007 | 0.428 | 0.152 | 0.276 | 0.032 | 0.923 | 0.505 | 0.224 |
| `dmcts` | 0.271 | 0.275 | 0.019 | 0.435 | 0.162 | 0.273 | 0.066 | 0.516 | 0.451 | 0.204 |

Read: the tuned PPO gains are not from fixing the item-use gap. They mostly
shift toward more face attacks and higher lethal-taking rate. The remaining
search gap still looks tactical/representational: PPO uses items roughly one
third as often as `dmcts` on this window and sees fewer lethal opportunities.

### Focused-opponent continuation

Tested whether the restored PPO benefits from continuing against specific pain
points instead of `mixed`:

```bash
uv run locma train --opponent max-guard --steps 100000 \
  --out runs/ppo-focus-maxguard-lr1e4-ent01-100k.zip --seed 31 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.01 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent max-attack --steps 100000 \
  --out runs/ppo-focus-maxattack-lr1e4-ent01-100k.zip --seed 32 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.01 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent dmcts:2,3,0,3 --steps 50000 \
  --out runs/ppo-focus-dmcts2x3-lr1e4-ent005-50k.zip --seed 33 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.005 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 4
```

Eval (`--games 150` for scripted baselines, `--games 60` for default `dmcts`,
seed `10000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | `dmcts` |
|--------|----------|--------|-----------|------------|-----------|---------|
| restored `ppo-shuffled-pool` | 0.610 | 0.637 | 0.560 | 0.573 | 0.590 | 0.233 |
| focus `max-guard` | 0.643 | 0.677 | 0.553 | 0.607 | 0.612 | 0.267 |
| focus `max-attack` | 0.640 | 0.683 | 0.540 | 0.580 | 0.601 | 0.275 |
| focus `dmcts:2,3` | 0.603 | 0.647 | 0.533 | 0.563 | 0.581 | 0.267 |

Read: focused heuristic-opponent continuation gives small hard-baseline gains but
does not reliably improve the target matchup it trained against. Training against
tiny fair search did not transfer enough to default `dmcts` to justify more
expensive search-opponent PPO training yet.

### Rich self-play / opponent mixture

Added weighted mixed-opponent specs:

- `mixed-rich:seed`: scripted/greedy/max-guard/max-attack plus low-budget fair
  `dmcts` variants;
- historical/pruned: `mixed-sota`, `mixed-rich-search`, and
  `mixed-rich-ladder` were tested but removed from code after they failed to
  produce robust gains over `mixed-rich`.

The learner is still a normal no-search PPO policy at inference. Search policies
only appear as training opponents.

Scout commands:

```bash
uv run locma train --opponent mixed-rich --steps 150000 \
  --out runs/ppo-richmix-lr1e4-ent01-150k.zip --seed 41 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.01 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent mixed-sota --steps 150000 \
  --out runs/ppo-sotamix-lr1e4-ent01-150k.zip --seed 42 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.01 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
```

Eval (`--games 150` for scripted baselines, `--games 60` for default `dmcts`,
seed `11000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | `dmcts` |
|--------|----------|--------|-----------|------------|-----------|---------|
| restored `ppo-shuffled-pool` | 0.563 | 0.667 | 0.520 | 0.527 | 0.571 | 0.258 |
| best tuned sparse | 0.580 | 0.707 | 0.573 | 0.583 | 0.621 | 0.258 |
| `mixed-rich` continuation | 0.597 | 0.700 | 0.580 | 0.567 | 0.616 | 0.267 |
| `mixed-sota` continuation | 0.587 | 0.700 | 0.563 | 0.570 | 0.611 | 0.250 |

Read: richer opponent mixtures help the scripted hard baselines and are
competitive with the best sparse-tuned PPO, but they still do not move the
default `dmcts` matchup. Adding cheating SOTA opponents did not beat the fair
rich mix, probably because some of the SOTA behavior depends on hidden state the
visible PPO cannot condition on. If continuing this line, prefer `mixed-rich`
over `mixed-sota` and increase fair-search weight/budget gradually.

Follow-up rich schedule commands:

```bash
uv run locma train --opponent mixed-rich-search --steps 150000 \
  --out runs/ppo-richsearch-lr1e4-ent01-150k.zip --seed 51 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.01 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train --opponent mixed-rich-ladder --steps 150000 \
  --out runs/ppo-richladder-lr1e4-ent01-150k.zip --seed 52 \
  --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.01 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
uv run locma train-schedule mixed-rich mixed-rich-search mixed-rich-ladder \
  --steps-per-phase 70000 --out runs/ppo-richschedule-lr1e4-ent01-210k.zip \
  --seed 53 --reward-mode sparse --learning-rate 0.0001 --ent-coef 0.01 \
  --init-model runs/ppo-shuffled-pool.zip --n-envs 8
```

Fast hard-opponent ranking (`--games 80`, `--dmcts-games 0`, `--workers 8`,
seed `14000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 |
|--------|----------|--------|-----------|------------|-----------|
| restored `ppo-shuffled-pool` | 0.581 | 0.631 | 0.562 | 0.588 | 0.594 |
| best tuned sparse | 0.588 | 0.631 | 0.544 | 0.625 | 0.600 |
| `mixed-rich` continuation | 0.575 | 0.669 | 0.588 | 0.575 | 0.610 |
| `mixed-rich-search` continuation | 0.581 | 0.650 | 0.594 | 0.594 | 0.612 |
| `mixed-rich-ladder` continuation | 0.575 | 0.650 | 0.575 | 0.619 | 0.615 |
| explicit rich schedule | 0.594 | 0.669 | 0.531 | 0.613 | 0.604 |

Small default-`dmcts` probe (`--games 50`, seed `14100000`):

| policy | `dmcts` |
|--------|---------|
| restored `ppo-shuffled-pool` | 0.320 |
| `mixed-rich` continuation | 0.260 |
| `mixed-rich-search` continuation | 0.230 |
| `mixed-rich-ladder` continuation | 0.300 |

Direct head-to-head vs restored PPO (`--games 200`, seed `14200000`):

| policy A | win rate vs restored |
|----------|---------------------:|
| `mixed-rich` continuation | 0.525 |
| `mixed-rich-search` continuation | 0.505 |
| `mixed-rich-ladder` continuation | 0.487 |

Read: richer fair self-play schedules can add a small hard-baseline bump, but the
effect is not robust head-to-head and did not improve the default `dmcts` gap.
The ladder version is the best of this follow-up on avg-hard3 and the least bad
new variant against `dmcts`, but it is not a promotion candidate over restored or
the prior rich mix. Continue opponent-mixture work only with a more targeted
diagnostic, for example item/lethal-heavy opponents or curriculum phases selected
by action-stat gaps.

## Guardrail checks before new training

These are cheap checks to rerun whenever the encoding, battle rules, or training
env changes.

1. **Encoding contract**

   ```bash
   uv run pytest tests/test_encode.py tests/test_env.py tests/test_registry.py tests/test_mcts.py
   ```

   Confirms semantic action round-trip, masks, readiness exposure, search-opponent
   training compatibility, and fair `dmcts` determinization.

2. **Artifact-aware PPO eval**

   ```bash
   uv run locma tournament random scripted greedy max-guard max-attack dmcts \
     ppo:runs/ppo-shuffled-pool.zip --games 200 --seed 1000000 --matrix
   ```

   Use held-out seeds. This requires the local PPO artifact; without it, this
   checkout cannot reproduce the PPO cells directly.

3. **Action-distribution drift**

   Record decision histograms for the current PPO, `balanced` heuristics, and
   `dmcts`: pass rate, summon rate, item rate, face attack rate, guard-clear rate,
   lethal-missed rate. If PPO is losing because of a narrow tactical blind spot,
   this is where it should show up.

## Experiments worth running

### A. Tactical feature ablation

Question: can a reactive net close part of the gap if the observation includes
small pieces of computed tactics instead of only raw card slots?

Add a second encoder variant with extra scalars:

- friendly reachable face damage this turn;
- opponent reachable face damage next turn, approximate and public-info only;
- friendly lethal available;
- own exposed-to-lethal flag;
- opponent Guard count and total Guard defense;
- friendly/opponent board attack, defense, and creature count;
- number of legal attacks, summons, and item uses;
- best one-action trade delta available.

Run:

```bash
uv run locma train-zoo --out runs/ppo-tactical.zip --seed 0 --obs-mode tactical
uv run locma tournament random scripted greedy max-guard max-attack dmcts \
  ppo:runs/ppo-shuffled-pool.zip ppo:runs/ppo-tactical.zip \
  --games 300 --seed 1000000 --matrix
```

Decision rule: tactical obs is interesting only if it improves both avg-hard3 and
PPO-vs-`dmcts`, not just one baseline matchup. A small hard-baseline bump with
flat `dmcts` means "better heuristic reflexes", not "search gap closed."

### B. One-ply oracle labels

Question: is PPO missing shallow tactics specifically, or is the hard part deeper
planning?

Historical/pruned: added `oracle1`, a deterministic one-ply teacher: score every legal action by
applying it once, resolving immediate terminal/lethal cases, and evaluating the
same board/health heuristic used by AZ-lite. This is intended as a possible
behavior-cloning teacher, not as a deploy target.

Teacher strength smoke (`--games 80`, 160 mirrored games, seed `5000000`):

| teacher | scripted | greedy | max-guard | max-attack | `dmcts:4,8,0,3` |
|---------|----------|--------|-----------|------------|-----------------|
| `oracle1` | 0.406 | 0.625 | 0.300 | 0.588 | 0.812 |

Read: the naive one-ply board/health oracle is too weak, especially against
`scripted` and `max-guard`, to use as-is for cloning. If revisiting one-ply
labels, improve the scorer first: explicit guard handling, lethal/anti-lethal,
mana spend, card draw, and item utility.

Code status: `oracle1` was pruned from the registry/code; this section is kept as
paper trail for the negative teacher result.

Expected read:

- high agreement and strong baseline play means shallow tactical labels are
  learnable;
- if it still loses to `dmcts`, the missing piece is deeper tree search;
- if agreement is low, the current flat observation/action net still struggles
  even with one-ply tactics, which points back to architecture.

### C. Search-guided play with PPO priors

Question: does the trained PPO contain useful priors for search, even if the raw
reactive policy is capped?

Use `puct-ppo`, which replaces `azlite`'s heuristic prior with the PPO policy
distribution over the 155 semantic slots while keeping the existing heuristic
leaf value. Compare at fixed simulation budgets:

```bash
uv run locma tournament greedy max-guard max-attack dmcts \
  azlite:25 puct-ppo:25,runs/ppo-shuffled-pool.zip \
  azlite:100 puct-ppo:100,runs/ppo-shuffled-pool.zip \
  --games 200 --seed 1000000 --matrix
```

This is the most direct bridge from PPO to AlphaZero-lite. If PPO priors improve
low-budget search, the net is useful as search substrate. If heuristic priors
still dominate, focus on value training or tactical features before building a
larger AZ loop.

### D. Learned value leaf

Question: can PPO's value head or a separately trained value model replace the
hand-written search leaf?

Train a value model on positions sampled from `dmcts` and baseline games, target
final outcome from the acting seat. Use it as the leaf evaluator in `dmcts`/PUCT,
with heuristic value as the control.

Decision rule: value leaf must improve search at equal simulation count, not only
imitate the final result offline. Offline explained variance alone is not enough.

### E. Token/attention feature extractor

Question: is the MLP losing because fixed slots hide relations?

Keep the semantic action space, but replace the flat MLP extractor with a custom
feature extractor over card tokens: hand, friendly board, enemy board, plus
segment/type embeddings. This is heavier than tactical scalars, so run it after
Experiment A unless A is clearly dead.

Decision rule: compare at equal wall-clock and equal env steps. A transformer that
needs much more compute for the same win rate is not a practical improvement.

Historical/pruned: implemented a lightweight `card-token` extractor that shares an MLP across the
20 base card slots, flattens encoded slot tokens, and combines them with scalar
context before the PPO policy/value heads.

First scout:

```bash
uv run locma train-zoo --out runs/ppo-cardtoken-zoo-200k.zip --seed 3 \
  --feature-mode card-token --steps-per-opponent 50000 --n-envs 8
```

Eval on the same no-search window (`--games 100` for scripted baselines,
`--games 50` for `dmcts`, seed `4000000`):

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | `dmcts` |
|--------|----------|--------|-----------|------------|-----------|---------|
| `ppo-cardtoken-zoo-200k` | 0.370 | 0.510 | 0.365 | 0.420 | 0.432 | 0.130 |

Training KL/clip fraction were high (`approx_kl` up to about 0.08,
`clip_fraction` about 0.4), so this first run mostly says the default PPO
learning rate is too aggressive for this extractor. Added `--learning-rate` and
reran the same scout at `1e-4`:

```bash
uv run locma train-zoo --out runs/ppo-cardtoken-zoo-200k-lr1e4.zip --seed 4 \
  --feature-mode card-token --learning-rate 0.0001 \
  --steps-per-opponent 50000 --n-envs 8
```

Eval on the same window:

| policy | scripted | greedy | max-guard | max-attack | avg hard3 | `dmcts` |
|--------|----------|--------|-----------|------------|-----------|---------|
| `ppo-cardtoken-zoo-200k-lr1e4` | 0.455 | 0.575 | 0.470 | 0.490 | 0.512 | 0.260 |

Read: lower LR fixed most of the instability and improved over the first token
run, but the 200k scout remains below restored PPO on hard scripted baselines.
It is not the next mainline unless a longer/lower-LR run shows a steeper learning
curve.

Code status: this branch pruned the `card-token` feature-mode implementation.
Main's tokenized observation/PPO2 path supersedes this narrower extractor scout.

## Experiments to avoid for now

- more plain PPO budget on the same env;
- larger vanilla MLPs;
- `VecNormalize`;
- health/board reward shaping;
- raw self-play league without search at play time;
- direct distillation of `mcts`/`dmcts` into the same reactive head.

Those have already been tested and were flat or negative.

## Recommended order

1. Recreate the current strongest PPO artifact if it is missing.
2. Run the artifact-aware tournament on held-out seeds.
3. Add action/tactical diagnostic histograms.
4. Try tactical scalars.
5. Try PPO-prior PUCT.
6. Only then invest in value-leaf training or token/attention architecture.
