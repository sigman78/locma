# PPO next experiments

_Date: 2026-06-27_

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

## 2026-06-27 implementation pass

Added experiment hooks without changing the deployed base model layout:

- `--obs-mode tactical` for PPO training, plus `ppo-tactical:path` for eval;
- `--reward-mode sparse|health|board`, defaulting to the old sparse reward;
- `--init-model path` warm-start for continuing PPO from a restored or distilled
  artifact;
- `dmcts:K,I,seed,turns,1` deterministic mode for behavior-cloning data;
- `locma action-stats POLICY --opponent OPP` for quick tactical histograms.

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

Create a practicum from a deterministic one-ply teacher: score every legal action
by applying it once, resolving immediate terminal/lethal cases, and evaluating a
public board/health heuristic. Behavior-clone it with the current semantic action
space.

Expected read:

- high agreement and strong baseline play means shallow tactical labels are
  learnable;
- if it still loses to `dmcts`, the missing piece is deeper tree search;
- if agreement is low, the current flat observation/action net still struggles
  even with one-ply tactics, which points back to architecture.

### C. Search-guided play with PPO priors

Question: does the trained PPO contain useful priors for search, even if the raw
reactive policy is capped?

Build a `puct` variant that replaces `azlite`'s heuristic prior with the PPO
policy logits over the 155 semantic slots, while keeping the existing heuristic
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
