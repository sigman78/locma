# PPO Ceiling Study — design

**Date:** 2026-06-30
**Status:** Approved (brainstorm) — pending implementation plan
**Owner artifact:** this spec + `docs/ppo-ceiling-study-plan.md` (the bite-sized plan)
**Execution note:** this study is designed to be **executed on a CUDA machine** (RTX
4080 16 GB box, or M1 Max via MPS) in a *fresh* Claude session, cloned from the
`feat/ppo-ceiling-study` branch. This `docs/` location is deliberate — the
`docs/superpowers/` tree is gitignored and would never reach the clone.

---

## Problem & objective

**Objective (locked):** *Settle the theoretical limit of the reactive PPO policy
alone* — answer, rigorously, whether the reactive net's ~0.60–0.64 avg-hard3 plateau
is the true ceiling or an under-tuning artifact.

**The prior (from `docs/worklog.md`, `docs/ppo-review.md` §8):** a long list of
training-method levers were each tested and came back **flat** — training budget
(100k→3M), `ent_coef`, obs richness, normalization, **net size**, both-seat, reward
shaping, self-play/league, and distillation. The breakthrough lever was **search at
play time** (`netdmcts` 0.817). The worklog's bottom line reads "every
imitation/training-method path is spent."

**The genuine gap this study closes:** those were mostly *one-factor-at-a-time*
probes, run with hand-rolled scripts and post-hoc A/B tables, with **no systematic
joint hyperparameter search**, **no in-training telemetry**, and only 3 of ~12 PPO
knobs ever exposed (`learning_rate`, `ent_coef`, `target_kl` — everything else rides
SB3 defaults). "We tuned a few knobs and it was flat" is **not** the same claim as
"a principled joint search found no headroom." This study makes the stronger claim
testable, with a symmetric decision rule so a **confirmed null is as valuable as a
lift**.

## Decisions locked during brainstorm

1. **Stay on stable-baselines3 (sb3-contrib MaskablePPO).** A different PPO
   implementation would *confound* the result (lift from the HP, or from the new
   algo?) and would change the model artifact, breaking the `netdmcts` `NetOracle`
   that loads the sb3 `.zip` — the project's strongest policy. Every number stays
   comparable to the existing worklog.
2. **PufferLib = a de-risk throughput spike only (Gate 0), not a migration.** Measure
   SPS, record a go/no-go for a *future* migration, but run the study on sb3
   regardless.
3. **Token net is the primary subject; flat net is a cheap control.** Token is the
   richest, currently-best reactive net (0.639) and the most under-tuned (only
   `LR`/`target_kl` ever set on it). Flat anchors comparability.
4. **Observation-encoding experiments are IN — as Phase 2**, run with Phase-1's best
   HPs, for clean HP-then-obs attribution.
5. **Symmetric, pre-committed decision rule: a +0.03 avg-hard3 lift over the B0
   baseline** (defined below), resolved with a 95% CI that excludes zero. Clears it →
   *headroom found*; whole search stays within ±0.03 → *ceiling confirmed robust to
   HP*.
6. **Search method: Optuna TPE + Hyperband pruning + fANOVA importance**, seeded with
   the worklog's known-good token HPs.
7. **Compute:** CUDA RTX 4080 (16 GB) box (primary) or M1 Max (MPS). The 4080's win is
   running **many search trials in parallel** (each net is tiny; 16 GB holds dozens);
   the flat MLP likely stays CPU-faster, the token transformer benefits from GPU.

## Non-goals (out of scope)

- **Beating search policies** (`dmcts`/`netdmcts`). The reactive→planning gap is a
  separate, structurally-different problem (search at play time). This study reports
  the best config's win-rate vs `dmcts` only as an *informational stretch gauge*.
- **Migrating training to PufferLib** (Gate 0 only informs a future decision).
- **Offline/imitation learning as a ceiling lever.** Behavior-cloning a search
  teacher is already shown to cap at ~0.40 agreement (`worklog.md` 2026-06-27). See
  "Offline-from-replays" below for the one bounded, optional role it keeps.
- A new RL framework, a new engine, or any change to the rules/forward model.

---

## Architecture — a 4-stage pipeline + one preliminary gate

```
Gate 0  PufferLib throughput spike ─► informs a FUTURE migration only; does NOT
          block or alter the study. Also picks device + n_envs for the 4080.

Stage 1  Telemetry            WinRateEvalCallback → TensorBoard (avg-hard3 live +
                              SB3 diagnostics). Doubles as the Stage-2 pruning signal.
Stage 2  Fast Bayesian sweep  Optuna TPE over the SB3 knobs; each trial trains a
                              token net to a reduced budget; Hyperband pruning kills
                              hopeless trials early; many trials parallel on the 4080.
Stage 3  Rigorous confirm     top-K survivors → retrain ×3 seeds @ full budget →
                              paired eval ~1000 games/opp/seed → tight CI.
Stage 4  Verdict + importance Apply the +0.03 rule vs B0; emit fANOVA HP-importance;
                              write the worklog verdict.
```

**Phasing** (one coherent study):

- **Phase 0 — infra.** PufferLib spike · expose **all** SB3 HPs + token-arch knobs
  through `train_agent`/`train_zoo`/CLI · build `WinRateEvalCallback` + TensorBoard.
- **Phase 1 — HP-only sweep** on the current token obs (V0): 1a core PPO knobs, 1b a
  focused arch sweep around the 1a winner → **verdict #1**.
- **Phase 2 — obs-encoding** variants V1/V2 with Phase-1's best HPs → **verdict #2**.
- **Flat control** — a small parallel sweep, only to confirm the flat ceiling does or
  doesn't move under the same treatment.

## The baseline: B0 (load-bearing)

"Default HP" is a trap: raw SB3 defaults (`LR 3e-4`, `target_kl None`) are
**known-broken for the token net** — the worklog shows they drive `approx_kl` to
~0.15 and the token net *degrades* with training. Beating a broken baseline by +0.03
would be a fake win.

**B0 = the worklog's established good token recipe:** `learning_rate = 1e-4`,
`target_kl = 0.025`, all other PPO knobs at SB3 defaults, token obs (V0), the fixed
zoo curriculum, `both_seat=True`, trained **from scratch ×3 seeds at the standard
800k-step curriculum budget (200k/opponent × 4)**. This is the honest ~0.60 bar. The
sweep must beat **B0** by +0.03 (→ ~0.63+). A confirmed null then means "the ceiling
resists tuning," not "we forgot to set the LR."

**Budget commensurability (pinned):** Stage-2 *trial* budget is **reduced** (~300–400k
total = ~75–100k/opponent) so the search is cheap and pruning ends bad trials at
~50–100k; Stage-3 *confirm* retrains the survivors **and B0** at the **full 800k**
budget so the verdict compares like-for-like. Never compare a reduced-budget trial
score directly to B0's full-budget score — promotion to Stage 3 re-trains at full
budget.

---

## Stage 1 — Telemetry (`WinRateEvalCallback`)

The missing piece that makes the whole study observable, and the pruning signal that
makes Stage 2 affordable.

- A `MaskableEvalCallback`-style sb3 callback that, every `eval_freq` steps (default
  **50 000**), plays a fixed paired set of games (default **120/opponent**) vs each of
  **{scripted, max-guard, max-attack}** with the **deterministic** policy on
  **held-out eval seeds** (the `1_000_000+` range, disjoint from training seeds).
- Logs to TensorBoard: `eval/avg_hard3`, `eval/vs_scripted`, `eval/vs_max_guard`,
  `eval/vs_max_attack`, alongside SB3's native scalars (`train/approx_kl`,
  `train/clip_fraction`, `train/entropy_loss`, `train/explained_variance`,
  `train/value_loss`, `train/loss`).
- The callback's `avg_hard3` at the final checkpoint is the **Optuna objective**;
  intermediate values feed the **pruner**.
- Cost: 120 × 3 × ~8 evals ≈ 2 880 *quick* (scripted-opponent, no search) games per
  trial — cheap and bounded.
- Set `tensorboard_log` to a per-trial run dir so curves are comparable across the
  study. (wandb is an optional later add; TensorBoard is zero-extra-dependency-risk
  and sufficient.)

---

## Stage 2 — The search space (Phase 1, token V0)

Two focused passes keep the joint dimensionality productive.

### 1a — core PPO knobs (arch fixed at current d_model=64 / n_layers=2 / n_heads=4 / features_dim=256)

| knob | Optuna distribution | note |
|---|---|---|
| `learning_rate` | loguniform [3e-5, 5e-4] | seed near 1e-4 (token sweet spot) |
| `target_kl` | categorical {0.02, 0.03, 0.05, None} | token needs a KL cap to not diverge |
| `n_steps` | categorical {1024, 2048, 4096} | rollout length — **never tested** |
| `batch_size` | categorical {64, 128, 256, 512} | **never tested** |
| `n_epochs` | int [3, 10] | **never tested** |
| `gamma` | categorical {0.99, 0.995, 0.999} | ~50-turn games → horizon may matter |
| `gae_lambda` | categorical {0.9, 0.95, 0.98} | **never tested** |
| `clip_range` | categorical {0.1, 0.2, 0.3} | **never tested** |
| `ent_coef` | loguniform [1e-3, 5e-2] | ~neutral on flat; retest on token |
| `vf_coef` | categorical {0.5, 1.0} | **never tested** |

Constraint guard: enforce `batch_size ≤ n_steps * n_envs` (skip/clip invalid combos —
SB3 requires the rollout buffer to be divisible into minibatches).

### 1b — architecture (small categorical sweep around the 1a winner)

`d_model {64,128}` · `n_layers {1,2,3}` · `n_heads {4,8}` · `features_dim {128,256}`,
with the 1a-best PPO knobs frozen. (The worklog found net *size* doesn't help the
flat net, but the token transformer's depth/width is genuinely untested.)

### Fixed across all trials

The existing **zoo curriculum** (`ZOO_OPPONENTS = greedy, scripted, max-guard,
max-attack`, back-to-back via `train_zoo`), `both_seat=True`, `n_envs` (a throughput
constant pinned to whatever maxes SPS on the box), a **reduced trial budget**
(~300–400k total steps; pruning ends bad trials at ~50–100k), held-out eval seeds.

### Search execution

- Optuna **TPE** sampler + **HyperbandPruner** (or MedianPruner), **SQLite storage**
  (resumable, parallel-worker-safe), `enqueue_trial` seeded with B0's known-good
  point so TPE doesn't re-derive it.
- Target ~**100–200 trials**, several parallel workers on the 4080, sized to overnight
  batches.

---

## Stage 3/4 — Eval & decision protocol

- **Metric:** avg-hard3 = mean win-rate vs {scripted, max-guard, max-attack},
  **deterministic** policy, **held-out** eval seeds.
- **Rigorous confirm:** the top-K (3–5) survivors **and** B0, each retrained at full
  budget × **3 seeds**, each evaluated **~1000 games/opponent/seed**.
- **Paired-difference resolution (the key to resolving 0.03):** evaluate B0 and each
  candidate on the **identical** eval-seed sets; compute the **per-seed win-rate
  difference** Δ; bootstrap a 95% CI **over the paired differences** (much tighter
  than differencing two absolute rates). The `draft-bench` paired machinery
  (`locma/harness/draft_bench.py`) and `play` Wilson-CI machinery are the reuse base.
- **Noise floor:** a B0-vs-B0 self-duel (different seed) via the existing
  `noise-floor` command must show measurement resolution **< 0.03** before any verdict
  is trusted.
- **Verdict (symmetric):**
  - `max_k (mean Δ_k) ≥ +0.03` **and** its 95% CI excludes 0 → **headroom found** —
    ship the recipe, update the training defaults, record it.
  - otherwise (every candidate within ±0.03 of B0) → **ceiling confirmed robust to
    HP** — a definitive null that retires the "maybe we just didn't tune it"
    hypothesis.
- **Stretch gauge (informational only):** the best config's win-rate vs `dmcts`
  (fair search), to contextualize the residual planning gap. Never part of the
  decision.

---

## Phase 2 — Observation-encoding variants

Run with Phase-1's best HPs (clean HP-then-obs attribution). Each is judged against
the **same +0.03 rule vs B0** (so obs and HP levers are measured on one ruler).

- **V0 — current token obs** (baseline): 17 per-token features + 13 tactical scalars
  (`TOKEN_FEATS=17`, `N_TACTICAL=13`, `encode.py`).
- **V1 — richer engineered scalars (cheap, additive):** extend the tactical-scalar
  block with derived tactics the flat-scalar A/B never included as *relations*:
  exposed-to-lethal (opp reachable face damage ≥ my health next turn), mana-left-after
  a greedy best-play, card-advantage (my hand+board − opp), my Guard count,
  turn-parity / on-the-play. Pure additions to the scalar vector; the extractor adapts
  via `N_TACTICAL`. ~+5 scalars.
- **V2 — relational trade matrix (heavier, optional/stretch):** a per-pair
  (my-attacker × op-blocker) relation block — can-A-kill-B (`a.atk ≥ b.def`), does-B-
  kill-A, favorable-trade — fed as a relation input through a small MLP and
  concatenated in the extractor. This is the `ppo-review.md` §8.4A "relational
  objects" lever. Specify in the plan; only build if V1 looks promising or the
  schedule allows.

---

## Gate 0 — PufferLib throughput spike

A throughput-only benchmark, **no learning**, time-boxed (~half a day):

- Measure steps/sec for **sb3 `SubprocVecEnv`** (n_envs ∈ {4, 8, 16}) vs **PufferLib
  native vectorization** wrapping `BattleEnv`, scripted opponent, fixed wall-clock.
- Also measure realistic *training* SPS (token update included) on **CPU vs 4080
  CUDA** to pick `device` + `n_envs` for the sweep.
- **Output:** a worklog table + a go/no-go note for a *future* Puffer migration.
  Does **not** block the study; the study is sb3 either way.

---

## Offline-from-replays (item d) — scoped

- **Out as a ceiling lever.** "Offline PPO" is really BC/offline-RL; behavior-cloning
  a search teacher already caps at ~0.40 agreement → PPO-level net (worklog
  2026-06-27). PPO is on-policy; replays cannot replace its fresh rollouts.
- **One bounded, optional role:** a **BC-warm-start → PPO-finetune** dimension
  (`warm_start ∈ {none, bc-greedy}`) using the *existing* `practicum`/`distill` infra.
  It might *speed convergence* (cheaper trials) even if it cannot raise the ceiling.
  **Deferred by default** — include only if the user opts in, to keep Phase 1 clean.
- Replays' other legit uses (eval fixtures, replay-determinism checks) are already
  covered by the existing harness and untouched here.

---

## Module layout / deliverables (all additive, behind flags)

**Modified (Phase 0):**

- `locma/envs/training.py` — `_make_model`, `train_agent`, `train_zoo` gain the full
  PPO knob set (`n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`,
  `clip_range`, `vf_coef`, `max_grad_norm`) + a `device` arg + token-arch kwargs
  (passed as `features_extractor_kwargs`) + an optional `callback`/`tensorboard_log`.
  `train_zoo` also gains `n_envs` (currently hardcoded to 1). Existing call sites keep
  working via defaults (byte-identical when unset).
- `locma/cli/app.py` — surface the new knobs on `train`/`train-zoo`; add the `sweep`,
  `ceiling-eval` (verdict), and `puffer-bench` commands.
- `pyproject.toml` — new `sweep` extra: `optuna>=3.6`, `tensorboard>=2.16` (SQLite is
  stdlib).

**New:**

- `locma/envs/eval_callback.py` — `WinRateEvalCallback` (Stage 1).
- `locma/envs/sweep.py` — Optuna TPE driver: config space, pruner, SQLite storage,
  parallel workers, objective = train token net + return `WinRateEvalCallback`'s
  final avg-hard3.
- `locma/harness/ceiling_eval.py` — rigorous paired-difference eval + bootstrap CI +
  the +0.03 verdict (reuses `draft_bench`/`play` machinery).
- `scripts/puffer_bench.py` — the Gate-0 throughput benchmark (recorded, not shipped
  as a CLI necessity).
- Phase 2: obs V1/V2 additions in `locma/envs/encode.py` (+ `extractor.py` for V2).

**Docs:**

- this spec + `docs/ppo-ceiling-study-plan.md` (the plan).
- `docs/worklog.md` — a Gate-0 SPS table, a Phase-1 verdict entry, a Phase-2 verdict
  entry.
- `docs/ppo-review.md` §8 — the ceiling verdict slotted into the open-levers section.

**Tests:**

- `WinRateEvalCallback` returns sane avg-hard3 on a tiny stub run; logs the expected
  TB keys.
- the sweep objective runs end-to-end on a 1-trial / tiny-budget smoke config and
  produces a finite score; SQLite study is resumable.
- `ceiling_eval` paired-difference + bootstrap CI is correct on synthetic inputs
  (known Δ, known CI); the +0.03 verdict branches both ways on crafted data.
- all new HP plumbing: a `train_agent(..., n_steps=…, batch_size=…)` smoke trains a
  few hundred steps and saves a loadable model; flat path stays byte-identical when
  no new knob is set.

---

## Assumptions & risks

- **Proxy↔rigorous correlation.** The Stage-1 in-training avg-hard3 (120 games/opp)
  must rank trials consistently with the Stage-3 rigorous eval, or pruning discards
  good trials. Mitigation: never *decide* on the proxy — the proxy only *prunes* and
  *ranks for promotion*; the verdict always uses the rigorous paired eval. Spot-check
  the proxy-vs-rigorous rank correlation on the first batch of survivors.
- **From-scratch vs self-play baseline.** The sweep trains from scratch; the 0.639
  self-play number is **not** the bar (B0's ~0.60 from-scratch is). Stated to prevent
  a moved-goalpost null.
- **Strong prior toward null.** The worklog predicts no lift. That is fine — the
  symmetric rule makes a clean, well-powered null a *publishable result*, not a
  failure. The risk is an *under-powered* null (too few games/seeds); the
  paired-difference protocol + noise-floor gate guard against it.
- **GPU non-determinism.** CUDA kernels are not bitwise-deterministic; this only
  affects training noise (already absorbed by multi-seed), not replay determinism
  (eval uses the deterministic policy + seeded engine). Keep eval on the seeded
  engine path.
- **`batch_size`/`n_steps` invalid combos** can crash SB3. Guard in the config space.

---

## Handoff procedure (this win32 box → the 4080 PC)

1. This session writes the spec + the implementation plan to `docs/` (tracked),
   creates branch **`feat/ppo-ceiling-study`**, commits both (+ a short
   `docs/ppo-ceiling-study-HANDOFF.md` pointer), and **pushes over HTTPS**
   (`git -c credential.helper='!gh auth git-credential' push ...`, per the repo's
   SSH-keyless push convention).
2. On the 4080 PC: `git clone` / `git fetch && git checkout feat/ppo-ceiling-study`.
3. Env: `uv sync --extra ml --extra dev --extra sweep` (the new `sweep` extra adds
   Optuna + TensorBoard). Verify CUDA torch sees the GPU (`torch.cuda.is_available()`);
   else fall back to MPS/CPU.
4. Start a fresh Claude session there; point it at `docs/ppo-ceiling-study-plan.md`
   and execute task-by-task (subagent-driven-development or executing-plans).
5. CI discipline carries over: `ruff check .` + `ruff format --check .` + `pytest -q`
   on `--extra dev`; format edits must be staged; use `uv run` for everything.

---

## Success criteria (what "done" looks like)

- A reproducible `locma sweep` that runs Optuna TPE with pruning + TensorBoard, on
  GPU, resumable from SQLite.
- A Gate-0 SPS table + Puffer go/no-go in the worklog.
- A **Phase-1 verdict** (headroom-found-with-recipe **or** ceiling-confirmed-robust)
  with the fANOVA importance map, decided by the rigorous paired +0.03 rule.
- A **Phase-2 verdict** on obs V1 (and V2 if built), on the same ruler.
- All code additive behind flags; the flat baseline and the `netdmcts` oracle
  untouched; the sb3 model artifact unchanged.
