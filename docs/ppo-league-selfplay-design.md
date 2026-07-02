# League (Fictitious) Self-Play for the Token PPO Net — Design

**Branch:** `feat/ppo-autoreg-action` (off `main`; this is a third experiment on
the same reactive-PPO exploration branch).
**Status:** approved design; execution intended for a **faster PC** (RTX 4080 box
or M1 Max) via the handoff doc. Tooling is CPU-testable; the ~3–4 hr training is
the reason to move it.
**Siblings:** the AR-head study (verdict: no-help) and the DeepSets-obs sketch on
the same branch.

## Objective

Turn the earlier throwaway 2-snapshot self-play probe into a proper, **tracked
league** (fictitious self-play): each round trains the current token net against a
per-episode mix of **all past frozen snapshots** + the ground baselines, then
snapshots itself back into the pool. Track `avg-hard3` per round. Then test the
best league net as a **netdmcts oracle** — where a stronger reactive net actually
pays off.

## Background / why this specific variant

Prior work (worklog 2026-06-27, "Self-play of token PPO2") already established:

- **Token net responds, flat net regresses.** Self-play *sharpened* the
  slot-addressable token net (+0.031, base 0.601 → r1 0.632) but *decayed* the
  flat net. So this study uses the **token** net.
- **Naive 2-snapshot self-play plateaus.** r1 +0.031 (real, ~2.6σ) then r2 +0.007
  (noise) → converged ~**0.639**, the project's strongest *reactive* net. It does
  **not** compound and stays far below search (netdmcts **0.817**).
- That probe mixed only the *latest* frozen self + baselines. **A league of all
  past selves (FSP) is the one untried variant** — it resists the
  forgetting/cycling that stalls 2-snapshot self-play (you must keep beating every
  earlier version, not just the last).

The reactive ceiling is structural (search is the lever); the honest value of a
stronger reactive net is as a **NetOracle for netdmcts**, whose current oracle is
`selfplay-r2` (0.639) → 0.817. Hence the oracle-downstream test is part of the study.

## Locked decisions

- **Net = token** (`obs_mode="token"`). Flat self-play is a known regression.
- **League = fictitious self-play (FSP), uniform.** Opponent pool each round =
  `[ppo:<snap> for every past snapshot] + [scripted, max-guard, max-attack]`,
  sampled uniformly per episode by `MixedOpponentPolicy`. The self:baseline ratio
  grows `r/(r+3)` as the league fills — a natural curriculum toward self-play.
- **6 rounds × 200k steps**, on an **800k token zoo-curriculum base** (round 0).
- **Conservative KL:** `target_kl=0.025` (the prior probe relied on it to keep
  self-play updates stable).
- **n_envs=1, DummyVecEnv, opponent built as a Python object** (see plumbing
  constraint). Continue the *same* model across rounds via `set_env` +
  `learn(..., reset_num_timesteps=False)`.
- **Oracle downstream:** best-`avg-hard3` snapshot → `netdmcts:8,40,1.5,<best>.zip`,
  eval `avg-hard3`, compare to 0.817.
- **Runs on the faster PC** (CUDA or MPS if available, else CPU); tooling is
  CPU-testable and handed off as a plan.

## The plumbing constraint (shapes the design)

The training opponent is normally rebuilt from a **spec string** inside each env
(so `SubprocVecEnv` can pickle it across processes). A league pool of snapshot
**paths** + baselines cannot ride a flat spec string (`ppo:<path>` already contains
colons; Windows paths contain backslashes). The registry `mixed` spec only builds
the *fixed baseline* pool — it has no snapshot support.

**Resolution:** the league loop runs **n_envs=1 (`DummyVecEnv`)** and constructs the
`MixedOpponentPolicy` **object directly** from a Python list of policies — no spec
round-trip, no subprocess. This is the same single-env setup the prior self-play
probe used. Single-env is slower but correct, and self-play is not env-throughput
bound the way from-scratch RL is.

## Pipeline

1. **Round 0 (base):** `train-zoo --obs-mode token`, 800k (200k × 4 curriculum),
   seed 0 → `round0.zip`. Eval `avg-hard3`; log as round 0.
2. **Rounds 1–6:** for each `r`:
   - `pool = [make_policy(f"ppo:{s}") for s in snapshots] + [make_policy(b) for b in baselines]`
   - `opp = MixedOpponentPolicy(pool, name=f"league-r{r}", seed=seed+r)`
   - `env = DummyVecEnv([partial(BattleEnv, opponent=opp, seed=seed+r, agent_seat=0, seat_random=True, obs_mode="token")])`
   - `model.set_env(env)`; `model.target_kl = 0.025`; `model.learn(200_000, reset_num_timesteps=False)`
   - `model.save(f"round{r}.zip")`; append to `snapshots`
   - eval `avg-hard3` → append CSV row; rewrite `league.csv`
3. **Oracle downstream:** pick `best = argmax_r avg_hard3(round_r)`; eval
   `netdmcts:8,40,1.5,<best>` `avg-hard3`; record vs 0.817.

## Success criteria (tracked, not a single ±0.03 gate)

- **Reactive:** does the league curve clear the prior plateau **0.639**, and does it
  *compound* across rounds or plateau again? Report the full per-round curve.
- **Downstream (the real payoff):** does the best league net as a NetOracle move
  netdmcts above **0.817**?

Both are reported with held-out-seed `avg-hard3`; a paired bootstrap (reusing the
`ar_study` helpers) quantifies the best-league-net vs the base and vs 0.639.

## Evaluation

`avg-hard3` = mean win-rate vs {scripted, max-guard, max-attack}, deterministic
policy, held-out eval seeds (`1_000_000+`), reusing `ar_study.hard3_per_seed`
(obs-agnostic — the token net is composed via `ppo:<snap>` → token encoder). The
league eval seeds are disjoint from any training seeds.

## Module layout

| File | Responsibility | ML dep |
|------|----------------|--------|
| `locma/envs/league.py` (new) | `league_pool_specs`, `build_league_opponent`, `_league_env`, `run_league`, CSV writer | yes (lazy) |
| `locma/cli/app.py` (modify) | `selfplay-league` command | no |
| `locma/harness/ar_study.py` (reuse) | `hard3_per_seed` for per-round eval | no (numpy) |
| `tests/test_league.py` (new) | pool-spec determinism, 2-round smoke league | mixed |

`league_pool_specs(snapshots, baselines) -> list[str]` is torch-free and pure
(`[f"ppo:{s}" for s in snapshots] + list(baselines)`), so pool composition is
unit-tested without the ML stack.

## Testing (TDD, CPU-verifiable)

1. **Pool specs:** `league_pool_specs(["a.zip","b.zip"], ["scripted"])` ==
   `["ppo:a.zip", "ppo:b.zip", "scripted"]`; length = `len(snapshots)+len(baselines)`.
2. **Opponent build:** `build_league_opponent(snapshots, baselines, seed)` returns a
   `MixedOpponentPolicy` whose pool length matches; seeded sampling is reproducible.
3. **Smoke league:** with a tiny pre-trained token base, `run_league(rounds=2,
   steps_per_round=300, eval_seeds=3)` returns 3 rows (round 0,1,2), writes
   `league.csv`, and creates `round1.zip`/`round2.zip` (pool grew).
4. **CLI:** `selfplay-league --help` lists the flags; a bad `rounds` is rejected.

## Assumptions & risks

- **CPU-only fallback.** If the faster PC lacks CUDA/MPS for torch, it still runs on
  CPU (the token net is small; self-play is single-env either way). The 4080 helps
  the token attention modestly, not dramatically.
- **Memory:** at round 6 the pool holds 6 snapshot nets + 3 baselines, each loaded
  lazily. Well within 16 GB.
- **Plateau is the likely outcome.** The reactive ceiling is structural; the league
  may only nudge past 0.639. That's still a valid, tracked result — and the oracle
  downstream is where a modest reactive gain could actually matter.
- **Not a NetOracle-format change** — the league net is a normal token MaskablePPO
  `.zip`, directly usable as a `netdmcts:` oracle (no format work).

## Execution outline (detailed in the plan + handoff)

1. Build + CPU-test `league.py` + CLI (Tasks 1–4), full-suite green (Task 5).
2. R0: train the 800k token base.
3. R1: run the 6-round league (`selfplay-league`), producing `league.csv` + snapshots.
4. R2: oracle downstream — best snapshot as a `netdmcts:` oracle, eval `avg-hard3`.
5. R3: record the curve + oracle result in `docs/worklog.md`.
