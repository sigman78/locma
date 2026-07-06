# E15 design: ranking-loss critic + AZ-v2 self-play (the "close-ish MuZero")

Status: DRAFT (pre-registered before any run). Branch feat/e15-ranking.
Author date: 2026-07-06.

## Motivation and lineage

Three results converge here:

- **E9** (ensemble distill, null): matching the ensemble's VALUES to 0.08
  RMSE did not match its ORDERINGS — the beam consumes sibling margins far
  finer than the regression residual. Escalation (a) was preserved verbatim:
  *pairwise ranking loss on actual beam-sibling pairs toward the ensemble's
  preference*, with the fidelity gate re-calibrated to sibling value GAPS
  instead of label variance. E15 Stage 1 is that retry.
- **E13** closed the training-data-diversity direction: the ~0.890
  single-critic ceiling needs a different signal KIND, not more data
  variety. Ranking loss is the recorded surviving lever.
- **E14a** located the reactive net's failure: turn-level branching (0.32
  disagreement at 2-4 legal actions vs 0.85 at 9-14), 18.3% missed forced
  wins, 3.3x item underuse — and refuted the intra-turn-context/obs story.
  Whatever training signal we aim at the policy head should target
  high-branching FIRST actions, and E14a's probe suite is the mechanism
  ruler for whether it landed.

MuZero framing: we already have MuZero-minus-learned-model (netdmcts pUCT +
priors + Dirichlet, record_selfplay, az_train soft-CE/MSE, azloop gated
iteration — netdmcts Phase 2, 2026-06-27). A learned dynamics model buys
nothing against a perfect ~ms/move simulator and is a NON-GOAL (see bottom).
What MuZero actually contributes here is its training loop shape: **policy
and value targets produced by search, consumed at scale, iterated.** The
assets that did not exist when the AZ loop ran in June: a ~100x cheaper
search operator (vbeam), a 0.926-strength teacher (the 3-critic ensemble),
`plan_turn(collect=)` backed-up targets, and the E7 process lesson (score
training-side artifacts UNDER THE PLANNER, not just reactively).

## Prior nulls this design must respect

| attempt | signal | verdict |
|---|---|---|
| self-play/league (June) | on-policy RL vs self | flat — reactive nets can't absorb planning |
| netdmcts Phase 2 AZ loop | visit-CE + outcome-MSE, 100 games/iter | null, no compounding |
| EXIT / vbeam distill (#62) | argmax-action CE (policy head) | null |
| E9 ensemble distill (#69) | ensemble-mean value MSE (critic) | null — values transfer, orderings don't |

Every prior attempt either used an argmax/MSE objective (ordering-blind) or
starved on data. E15 changes the objective to ordering-aware and the data
scale by ~10-100x. If it still nulls, the "reactive nets can absorb
planning" question is closed for this kit.

## Stage 1 — ranking-loss critic distillation (the load-bearing experiment)

**Question:** does an ordering-aware objective transfer the ensemble's
sibling-ranking ability into ONE critic (1/3 evaluator compute, and the
first training-side result past the 0.890 single-critic ceiling)?

**Data.** Extend `plan_turn`'s collect tuple with the action prefix
(additive 5th field; two in-repo consumers updated) so sibling groups are
reconstructable: a group = states the beam actually sorted against each
other at one (plan call, depth). Re-run `collect_ensemble_data`-style
harvesting with group ids: vbeam-on-ensemble self-play vs the E9 opponent
mix, both seats, ~3x E9's volume (target ~100k kept states, ~300k usable
same-group pairs). Labels per state: ensemble MEAN value (as E9) — the pair
label is its sign of difference. Keep per-pair ensemble margin
|target_i - target_j| for gating and loss weighting. Cost: ~2-3 h on 19
workers (ensemble vbeam ~2 s/game) + minutes of batched labeling.

**Loss.** RankNet-style pairwise logistic on same-group pairs,
margin-weighted, PLUS a small MSE anchor to the ensemble mean
(lambda ~0.25): `plan_turn` ranks completed plans ACROSS depths and mixes
critic stops with win/loss sentinels, so absolute calibration in [-1,1]
must not drift while orderings sharpen. Critic-branch-only fine-tune of
each `depot:shared/shared_sX` (reuse `vbeam_fvi.train_value_head`
scaffolding with the loss swapped; frozen shared extractor, policy path
byte-identical — same conditions as E9 so the objective is the only moved
variable).

**Pre-registered gates.**

- **G1 fidelity (fixes E9's mis-calibrated gate):** on held-out sibling
  pairs, ordering accuracy vs ensemble labels. Base single critic accuracy
  = a0 (measured on the same pairs). Gate: FT accuracy >= a0 + 0.5*(1 - a0)
  (closes at least half the ordering gap). Report margin-stratified
  accuracy (pairs with ensemble margin above/below the E9 residual 0.08).
- **G2 primary:** paired 40x25, `vbeam:vrank_sX` vs `vbeam:depot:shared_sX`
  at fresh 10M seeds; CI-positive fulls confirmed at 11M. PASS iff full AND
  confirm ci_lo > 0.
- **G3 compression/promotion:** `vbeam:vrank` (single) vs the ensemble RoR
  (0.926). Promote vrank to planner recipe of record iff G2 passes AND the
  vs-ensemble CI contains or exceeds zero (non-inferior at 1/3 compute).
  Ensemble(vrank x3) vs ensemble RoR runs as a bonus arm iff G2 passes.

**Kill criteria (decisive either way).**

- G1 FAILS -> the frozen shared extractor cannot represent the ensemble's
  ordering (E9's untested capacity co-explanation confirmed). Record
  "unfreeze extractor" as the only remaining escalation; do not run G2.
- G1 passes, G2 null -> in-distribution orderings learned but play does not
  move; the critic-distillation direction is closed outright (third
  formulation, third null, no surviving objection).

## Stage 2 — AZ-v2 policy round (the MuZero policy-improvement side)

Runs after Stage 1 regardless of its outcome (shares the harvested data;
the best available critic — vrank if G2 passed, else shared — generates it).

**Signal:** listwise soft-CE on the policy head toward
softmax(backed-up sibling values / tau) at ROOT decision points — the
visit-distribution analog, soft and margin-aware where EXIT's argmax CE was
brittle. Sample weighting toward what E14a indicts: weight by legal-action
count (branching), which also concentrates mass on first actions. Reuse
`vbeam_distill.train_policy_head` scaffolding (frozen features, critic path
byte-identical) with the loss swapped.

**Pre-registered gates (paired 40x25, 10M/11M):**

- G4 reactive: `vrankpi_sX` vs `depot:b0k`. PASS iff full+confirm
  ci_lo > 0 — this would be the first reactive absorption win in five
  attempts; expectation is null.
- G5 mechanism (informational, cheap): re-run the E14a probe suite on the
  new net — missed-lethal rate, item underuse ratio, first-action
  disagreement, per-turn regret. Directional movement here with a null G4
  is still a finding (signal lands but does not cash out).
- G6 planner side-effects: `vbeam:vrankpi` vs its base — the policy head
  feeds vbeam only through would_pass stop-eligibility; verify no
  regression (ci_hi > -0.01 non-inferiority).

## Stage 3 — iterate (only if Stage 1 G2 or Stage 2 G4 passes)

Re-harvest with the adopted net(s) in the evaluator, repeat the passing
stage: 2-3 iterations under `azloop`'s composite adoption gate (h2h vs
parent AND no avg-hard3 regression, keep-best, early-stop). Fresh seed
sub-ranges from 12M. Gumbel-style root selection for netdmcts is deferred
until something compounds — infra polish is worthless if one round nulls.

## Budgets

| item | cost |
|---|---|
| collect extension + consumers + tests | half a day |
| Stage 1 data harvest | ~2-3 h, 19 workers |
| Stage 1 FT (3 seeds) | minutes (frozen features) |
| G2/G3 verdicts + confirms | ~4-6 h eval |
| Stage 2 FT + G4/G5/G6 | ~3-4 h |
| driver, worklog, PR | half a day |

Total: one build day + one overnight. Seed ledger: 10M primary, 11M
confirm, 12M+ reserved for Stage 3 iterations (1M-9M already spent:
standard/E8/E11/E12x2/exploit/E13x2/E14a).

## Non-goals

- Learned latent dynamics / search in latent space (true MuZero): redundant
  vs a perfect fast simulator; weeks of work; would also need Stochastic
  MuZero machinery for LOCM's draws and hidden hand. Determinization
  (dmcts K) already covers hidden info at the search layer.
- Replay buffer / reanalyze: only relevant if Stage 3 compounds.
- Architecture changes (unfreezing the extractor, LSTM, bigger trunks):
  E6 killed recurrence; extractor unfreeze is recorded as the post-G1-fail
  escalation only.
- Obs-space changes: killed by E14a.

## Artifacts

Driver `scripts/e15_driver.py` (idempotent stages, smoke via E15_SMOKE=1,
gates recorded machine-readable as `e15_gates`), data
`runs/e15-rankdata-*.npz`, models `runs/vrank_s{0,1,2}.zip` /
`runs/vrankpi_s{0,1,2}.zip`, results `runs/e15-summary.json`, log
`runs/e15-overnight.log`. No depot publish without an explicit promotion
decision.
