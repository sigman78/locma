# Flat-obs PPO with Autoregressive Action Head — Design

**Branch:** `feat/ppo-autoreg-action` (off `main`).
**Status:** approved design; execution runs inline on this machine (CPU torch).
**Sibling:** `feat/ppo-ceiling-study` (HP-sweep probe). This is an independent
direction — a different *policy architecture*, not an HP or obs variant.

## Objective

Settle, with a symmetric +0.03 paired-difference verdict, whether **factoring the
action head autoregressively** — while holding the observation and action space
fixed — moves the reactive PPO policy's `avg-hard3` above the flat 155-way softmax
baseline, or confirms that the flat masked categorical already captures whatever
the reactive policy can express.

**Hypothesis.** The flat 155-way masked softmax has to learn each `(type, source,
target)` cell largely independently. An autoregressive factorization shares credit
across symmetric slots: the `type` head learns "attacking is good now" once instead
of across 42 attack cells; the `target` head learns "hit face when lethal" once
across all attackers. That sharing could improve sample efficiency and the reachable
policy — or it could be null, which is an equally valid, publishable result under
the symmetric bar.

## Locked decisions

- **Single lever = the action head.** Observation stays exactly `encode_battle`
  (flat-308 `Box`). Action space stays exactly `Discrete(155)` with today's exact
  legality mask. Only the action *distribution* changes. This isolates the head as
  the sole independent variable and keeps the result directly comparable to the
  flat baseline.
- **True conditional autoregressive**, not factored-independent. Target legality
  depends on the chosen source, so independent per-factor masks would leak
  probability to illegal combinations and confound the test. We sample
  `type → source|type → target|type,source`, each masked given the prefix.
- **`Discrete(155)` retained end-to-end.** Actions in the rollout buffer stay flat
  ints; masks stay 155-bool. MaskablePPO's rollout / GAE / buffer / mask plumbing is
  untouched. We swap only the policy's head and its distribution. The saved artifact
  stays a normal `MaskablePPO.load`-able `.zip`.
- **Baseline = today's flat recipe**, same budget/seed (see Baseline below). The AR
  head must beat it by **+0.03 `avg-hard3`, symmetric, pre-committed**, resolved by a
  paired-difference bootstrap CI over shared eval seeds.
- **Runs inline on this machine.** Torch is CPU-only here; flat-obs PPO is
  env-step-bound, not matmul-bound, so CPU + `n_envs` parallelism is the expected
  path. No GPU handoff.
- **Out of scope:** NetOracle / netdmcts integration (the AR policy is not a
  drop-in oracle), obs-encoding changes, token+pointer variants (noted as a future
  stretch only), HP sweeps (that is the sibling study).

## The action factorization

From `locma/envs/encode.py`, the 155 indices already partition along
`type → source → target`:

| type   | index range | source            | target            |
|--------|-------------|-------------------|-------------------|
| PASS   | 0           | —                 | —                 |
| SUMMON | 1–8         | hand slot 0–7     | —                 |
| USE    | 9–112       | hand slot 0–7     | target code 0–12  |
| ATTACK | 113–154     | board slot 0–5    | target code 0–6   |

Reconstruction is pure integer arithmetic:

```
decode(idx):
  idx == 0            -> (PASS,   0,            0)
  1  <= idx <= 8      -> (SUMMON, idx-1,        0)
  9  <= idx <= 112    -> (USE,    (idx-9)//13,  (idx-9)%13)
  113<= idx <= 154    -> (ATTACK, (idx-113)//7, (idx-113)%7)

encode(type, source, target) = base[type] + source*ntarget[type] + target
  where base   = {PASS:0, SUMMON:1, USE:9, ATTACK:113}
        ntarget= {PASS:1, SUMMON:1, USE:13, ATTACK:7}
```

Head widths are fixed at the maxima — `type`(4), `source`(8), `target`(13) — and
masking zeroes the out-of-domain entries (ATTACK uses only sources 0–5, targets
0–6; SUMMON/PASS use only target 0). One `source` head and one `target` head serve
all types.

`decode`/`encode` and `factor_masks` are torch-free and live in a new
`locma/envs/action_factor.py`, unit-tested independently of the ML stack.

## The autoregressive distribution

**Conditioning.** From the shared policy latent `z` (SB3's `mlp_extractor.policy_net`
output):

```
type_logits   = W_type(z)                                    # (B, 4)
source_logits = W_source(concat(z, emb_type(type)))          # (B, 8)
target_logits = W_target(concat(z, emb_type(type), emb_source(source)))  # (B, 13)
```

`emb_type` (4→d) and `emb_source` (8→d) are small learned embeddings.

**Masking — single source of truth.** All conditional masks are *derived* from the
existing flat-155 mask, so legality can never diverge between the head and the env:

- `type_mask[t]  = OR` of the flat mask over type `t`'s index segment.
- `source_mask[t][s] = OR` of the flat mask over source `s`'s target row within type `t`.
- `target_mask[t][s] = ` that row of the flat mask.

Because each conditional is masked to legal continuations, the reconstructed flat
index is **always legal** — the head cannot emit an illegal action, the same
guarantee the flat categorical has today.

**Scoring.**

- `log π(a) = log π(type) + log π(source|type) + log π(target|type,source)`, summing
  only the heads that apply: PASS → 1 term, SUMMON → 2, USE/ATTACK → 3.
- `entropy = Σ` conditional entropies — the exact joint entropy of a valid
  autoregressive factorization, so `ent_coef` retains its meaning.

**Three code paths, one distribution:**

- *Rollout* (`forward`): sample `type`, then `source|type`, then `target|type,source`,
  each masked; reconstruct the flat int; return `(action, value, log_prob)`.
- *Update* (`evaluate_actions`): decode the stored flat action into
  `(type, source, target)` tensors, compute the three conditional log-probs by
  teacher forcing (vectorized), sum; return `(value, log_prob, entropy)`.
- *Deterministic play* (`predict`): argmax at each step under the mask.

A unit test asserts the teacher-forced path and the sampling path agree on log-prob
for the same `(obs, action, mask)`.

## Wiring into MaskablePPO

Minimal surgery, chosen specifically to keep comparability:

- Action space **stays `Discrete(155)`**; masks stay 155-bool; the buffer stores
  flat ints. MaskablePPO's rollout/GAE/buffer/mask code is unchanged.
- A custom `MaskableAutoregressivePolicy` (subclass of `MaskableActorCriticPolicy`)
  replaces the single 155-logit head with the three conditional heads + embeddings.
  The **critic is unchanged** (value head on `latent_vf`).
- A custom `MaskableAutoregressiveDistribution` implements the three code paths above.
- `_make_model` gains a `head` selector: `"flat"` → today's `MlpPolicy` (byte-identical
  baseline), `"autoreg"` → `MaskableAutoregressivePolicy`. Obs stays flat either way.

The saved model is a normal MaskablePPO `.zip`; `MaskablePPO.load` reconstructs the
custom policy because the class is importable in-package (covered by the smoke test).

## Training & baseline

- **Baseline (B_flat):** `head="flat"`, LR 3e-4, ent_coef 0.02, both-seat, zoo
  curriculum `(greedy, scripted, max-guard, max-attack)` at 200k steps each
  (800k total), seed 0. This is exactly today's `train-zoo` default.
- **Candidate (B_ar):** `head="autoreg"`, *identical* recipe, budget, curriculum,
  and seed. The only difference is the head.
- `--head {flat,autoreg}` added to `train` and `train-zoo`.

## Telemetry ("observe effectiveness")

- Per-head entropy (type / source / target) recorded to the SB3 logger every rollout,
  to watch whether the factorization is learning structure (e.g. the type head
  sharpening while target stays exploratory).
- A lightweight periodic eval callback runs `avg-hard3` every K steps and records it,
  giving a learning curve rather than only an endpoint.
- Sink: SB3 logger + a CSV dump. No TensorBoard dependency added.

## Evaluation & decision rule

- **Metric:** `avg-hard3` = mean win-rate vs {scripted, max-guard, max-attack},
  deterministic policy, over a held-out eval-seed set (1_000_000+ range) shared
  between B_flat and B_ar (common random numbers → paired samples).
- **Verdict:** paired-difference bootstrap CI on `avg-hard3(B_ar) − avg-hard3(B_flat)`
  against a **symmetric, pre-committed ±0.03** bar:
  - point estimate ≥ +0.03 **and** CI excludes 0 → the AR head is a real lever.
  - the whole CI within ±0.03 → factoring does not move the reactive ceiling
    (ceiling-confirmed for the head dimension).
  - otherwise → inconclusive at this budget; report and stop (no budget-chasing).
- Stats are a small pure-numpy helper (`paired_bootstrap_ci`, `decide`) in
  `locma/harness/ar_study.py`, with `avg_hard3_per_seed` + `run_verdict` runners and a
  `locma ar-eval` CLI.

## Module layout

| File | Responsibility | ML dep |
|------|----------------|--------|
| `locma/envs/action_factor.py` (new) | `decode`/`encode`, `factor_masks(flat_mask)` | no |
| `locma/envs/ar_distribution.py` (new) | masked autoregressive distribution | yes |
| `locma/envs/ar_policy.py` (new) | `MaskableAutoregressivePolicy` (3 heads + critic) | yes |
| `locma/envs/ar_callbacks.py` (new) | per-head entropy + periodic `avg-hard3` eval | yes |
| `locma/envs/training.py` (modify) | `head` param + wiring; CLI thread-through | yes (lazy) |
| `locma/harness/ar_study.py` (new) | avg-hard3 + paired bootstrap + verdict + runner | no (numpy) |
| `locma/cli/app.py` (modify) | `--head` on train/train-zoo; `ar-eval` command | no |
| `locma/policies/ppo.py` (verify) | `load`+`predict` round-trips the custom policy | yes (lazy) |
| `tests/…` | round-trip, mask reconstruction, distribution props, smoke train | mixed |

## Testing (TDD, CPU-verifiable)

1. **Round-trip:** `encode(decode(idx)) == idx` for all 155; decode agrees with
   `sem_index` semantics on sampled concrete actions.
2. **Mask reconstruction:** for random legal sets, the union of legal
   `(type, source, target)` under the derived conditional masks equals the flat mask
   exactly (no illegal admitted, none legal dropped).
3. **Distribution properties:** `log π(a) = Σ` conditional log-probs; sampled actions
   are always legal; `entropy = Σ` conditional entropies; teacher-forced and sampled
   log-probs agree; masked probabilities sum to 1 over the legal set.
4. **Policy forward:** output shapes; deterministic argmax is legal; batch
   `evaluate_actions` runs and returns finite grads.
5. **Smoke train:** 200-step `MaskablePPO.learn` end-to-end with the AR policy;
   `save`→`load`→plays a full legal game; artifact is a standard `.zip`.

## Assumptions & risks

- **Custom distribution correctness** is the main risk — retired by CPU unit tests
  (properties 1–4) before any training run.
- **`load` needs the policy class importable** — it is in-package; smoke test covers
  the save/load round-trip explicitly.
- **CPU training time** — 800k × 2 runs on CPU; mitigated by `n_envs` parallelism.
  The net is tiny; the cost is the Python game-sim in the env step, same as the
  existing flat baseline.
- **Null result is success**, not failure — the symmetric bar makes "the flat head
  already captures the reactive policy's expressiveness" a real conclusion.
- **Not a NetOracle drop-in** — deliberately out of scope; this probes the standalone
  reactive policy only.

## Execution outline (detailed in the plan)

1. Build + CPU-unit-test `action_factor`, `ar_distribution`, `ar_policy` (TDD).
2. Wire `head` into training + CLI; smoke train + save/load round-trip.
3. Build `ar_study` stats + `ar-eval` CLI; telemetry callback.
4. Train B_flat and B_ar (identical recipe/budget/seed).
5. Run the paired verdict; write the result into `docs/worklog.md`.
