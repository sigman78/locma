# PPO review — feedback: hypothesis for the plateau and ranked next experiments

_Date: 2026-07-01. Follow-up to `docs/ppo-review.md` (the action-space review) and the
ceiling study (`docs/ppo-ceiling-study-design.md`, R1–R4 in `docs/worklog.md`)._

## 0. What has been tried (and what it bought)

| Lever | Result |
|---|---|
| Semantic (structured) action space | The one big win: 0.28 → 0.45 avg, foundation of everything since |
| Tokenized obs + self-attention (PPO2) | Parity to +0.02 — real but within seed noise, 4x cost |
| Distilling from MCTS | Marginal (~0.40 agreement cap; teacher-student info gap with the cheating teacher) |
| League / self-play | +0.03 front-loaded, plateaus ~0.64, no compounding |
| Hyperparameter tuning (Optuna, R2/R4) | Null; best sweep candidate *regressed* -0.04 at full budget |
| Reward shaping, obs richness, net size, normalization, both-seat | All flat (ruled out) |
| Search at play time (`netdmcts`, same net as oracle) | **+0.18** (0.639 → 0.817), the only lever that moved since the action-space fix |

Reactive ceiling: **avg-hard3 ≈ 0.62–0.66**. Search over the *same* net: **0.82**.
That asymmetry is the whole story, and the hypothesis below starts from it.

---

## 1. Hypothesis

### H1 (primary): the residual gap is decision-time computation, not representation, data, or optimization

Every training-side lever is flat, while wrapping the *identical* net in PUCT adds
+0.18 — and even a net-free searcher (`azlite`, heuristic oracle) reaches 0.74,
above every reactive net ever trained here. The stronger policies are stronger
because they *compute at the decision point*; the reactive net must amortize that
computation into a single forward pass. PPO can only learn the average-case reflex
over a stochastic, partially observed opponent distribution; it cannot condition on
the exact combinatorial consequences of this turn's action sequence. More RL of any
flavor (self-play, league, HP, reward) adds data to the same estimator — it cannot
add the missing computation. This predicts exactly what was observed: all
training-method levers flat, search-method levers large.

### H2 (refinement of H1): the *specific* missing computation is within-turn plan composition

A LOCM turn is a *sequence* of atomic actions (summons, items, ordered attacks)
whose value realizes only at end of turn; the atomic-action MDP makes PPO pick each
step greedily under a learned preference, so plans that require a locally neutral
or bad first step (kill the Guard with the weaker attacker *so that* the stronger
one goes face; play the item *before* attacking) are systematically hard to
represent. Crucially, **the engine is deterministic and uses no hidden information
within your own turn** (draws happen only at `start_turn`, `battle.py:72`), so
own-turn lookahead is *fair* and cheap — this is the slice of `netdmcts`'s +0.18
that a near-reactive policy could legally keep. Champion LOCM bots are exactly
this shape: turn-plan enumeration + a state evaluator.

(The autoregressive-action-space attempt is consistent with H2's diagnosis but
attacks it from the representation side only — the head can *express* a joint plan,
but it still gets no search over consequences, so a marginal result is expected.)

### H3 (secondary): partial observability caps the reactive ceiling from above

The obs is memoryless: opponent hand is a count (`engine.py:56`), and nothing
carries history (what they held back, what they've seen). Two states with identical
obs can have very different futures, and a reactive policy must play the average.
Search partially escapes this by determinizing (sampling K worlds); a reactive net
cannot. Part of the 0.62→0.82 gap is therefore *information*, not just planning —
and it bounds how far even perfect distillation of `netdmcts` can go (the teacher's
visit distribution is itself a function of sampled hidden worlds, so it is not
realizable by any deterministic function of the public view; the old ~0.40
agreement cap is partly this, not only the cheating-teacher gap).

### H4 (hygiene): part of the observed optimization pathology is self-inflicted

Code inspection (section 3) found a likely-real training-noise source — **dropout
inside the PPO ratio** — that plausibly explains the token net's documented
`approx_kl` ≈ 0.10–0.15 blowup at default LR, i.e. the tuned recipe
(`lr=1e-4, target_kl=0.025`) may be *compensating for a bug* rather than
expressing a true optimum. Plus a real bug in the brand-new token-v1 threat
scalars that would mostly nullify the variant R5 is about to test. Fixing these
is nearly free and must precede any new verdicts.

---

## 2. Future experiments, ranked by complexity

Cheapest first. E1–E2 are bug-driven and should land before R5 runs; E4–E5 are the
directions with real headroom under H1/H2; E6+ are progressively speculative.

### E1 — trivial: remove dropout from the PPO path

Set `dropout=0.0` in `TokenSetExtractor` (or plumb it and default the *training*
path to 0). Re-run the B0 recipe x3 seeds; additionally retry `lr=3e-4` /
`target_kl=None`, which the worklog declared broken — if H4 is right, the
"token nets need gentle PPO" conclusion may partly dissolve, and the true HP
optimum shifts. Cost: ~1 line + one B0-scale rerun (~75 min for 3 seeds at the
observed pace). Verdict via the existing `ceiling-eval` ruler.

**Outcome (2026-07-02, `fix/ppo-e1-e3`): tested and REVERTED — dropout is a
feature, not a bug.** Dropout-0 at the B0 recipe regressed -0.037 [-0.044,
-0.030] paired vs B0. The N-battery decomposed it: the isolation arm (dropout
0.1 on fixed code) sits at -0.009 vs B0, so ~-0.028 is dropout removal itself;
and lr=3e-4 *without* dropout still collapses (-0.247 uncapped, -0.022 with the
KL cap), refuting the "gentle recipe was compensating for dropout" hypothesis —
the token net's high-LR instability is intrinsic. The ratio-contamination
mechanism is real but its cost is measurably dominated by the regularization
benefit at this scale. Default restored to 0.1 with a pinning test; H4 is
largely refuted on the dropout front.

### E2 — trivial: fix the token-v1 threat scalars, then run R5

`op_reachable` currently filters opponent creatures by `can_attack and
not has_attacked` (`encode.py:304-306`) — stale info that *undercounts* the threat
(section 3, finding 2). Fix to sum over the whole opponent board, regenerate
nothing (env-computed), rerun the planned R5 obs-v1 experiment on the same +0.03
ruler. Without the fix, R5's verdict on "does threat awareness help" is invalid —
`exposed_to_lethal` is frequently 0 in genuinely lethal positions, which is worse
than absent (a confidently wrong feature).

**Outcome (2026-07-02, `fix/ppo-e1-e3`): fixed; final verdict null.** On the
dropout-0 arms V1-fixed showed a small CI-clean +0.008 over V0, but at the
winning recipe (dropout 0.1) the R5-proper paired eval vs B0 came back +0.003
[-0.004, +0.010] — indistinguishable from V0. Phase-2 (obs-encoding) verdict:
ceiling-confirmed. The correctness fix stays (a confidently wrong feature is
still worse than a neutral one); the threat scalars are not a lever.

### E3 — easy: measurement/diversity hygiene (protects conclusions, not a win-rate lever)

- `WinRateEvalCallback._on_step` modulus can silently never fire (section 3,
  finding 3) — gate on bucket crossing instead.
- Parallel envs replay overlapping episode seeds (finding 4) — stride per-env
  seeds.
- `BattleEnv.reset` never reseeds/resets the opponent policy (finding 5) —
  matters if a stochastic opponent (`mixed`, `mcts`) is ever trained against.

### E3a — easy: tactics diagnostic suite (measure the sequencing failure before fixing it)

Directly targets H2's concrete symptom: within-turn plan composition (buff a
creature *before* attacking face, kill the Guard with the weaker attacker, item
before combat). Mine `netdmcts` games for decision points where the teacher's
chosen line is a multi-step sequence (item→attack, buff→lethal, guard-break→face)
and replay the raw reactive net on those positions. Output: a tracked metric —
"finds buff-lethal X%", "breaks guard then goes face Y%" — that every later
experiment (E4, E5) reports against, instead of arguing from avg-hard3 alone.
The practicum/replay infra covers most of this; ~a day of work, no training.

Why this matters mechanically: the buff action's payoff reaches PPO only through
the value bootstrap (`advantage(buff) ≈ V(after buff) − V(before)`), and the token
obs *does* expose the buffed stats and ready bit — so failures here localize the
problem to the value head's sharpness, which is exactly what E4's value
distillation and E5's planner attack. A tactics suite tells you which.

### E4 — medium: distill `netdmcts` (the fair 0.817 teacher) properly — soft targets + DAgger

The old distillation result ("caps ~0.40 agreement") predates the current best
teacher and used the *cheating* `mcts:100`, whose decisions depend on hidden
state the student cannot see. `netdmcts` is a function of the public view +
sampled worlds — a categorically better teacher. Three upgrades over the old BC:

1. **Soft targets**: train on the PUCT *visit distribution* (already recorded by
   `record_selfplay`), not top-1 actions — much denser signal, and exactly what
   worked for AlphaZero policy heads.
2. **DAgger loop**: collect teacher labels on the *student's* trajectories
   (fixes BC's compounding state-distribution shift — the student visits states
   the teacher's games never reach).
3. **Value distillation**: regress the student's value head on the search values
   (also already in the AZ pipeline) — needed anyway if the student ever seeds a
   PPO finetune or a `NetOracle`.

Expected: this is the honest measurement of "how much of search can be amortized."
H3 predicts it will *not* reach 0.82; anything ≥ 0.70 would be the strongest
reactive net by a wide margin. Cost: generation is the expensive part but
`record_selfplay` + `distill.py` are 90% of the infra; DAgger is a loop around them.

### E5 — medium: learned-value own-turn search ("planning-lite")

The direct attack on H2. At decision time, enumerate (beam-search) own-turn action
*sequences* on a cloned state — legal, fair, deterministic (no draws, no hidden
info within the turn) — and score the end-of-turn afterstate with a learned value
function (the AZ/PPO value head, or E4's distilled one); execute the best
sequence's first action (or the whole sequence). This is 1-turn depth, no opponent
model, no determinization — 10–100x cheaper than `netdmcts`'s K×I=320 sims, and it
captures precisely the within-turn compositionality a reactive net lacks. `azlite`
already demonstrates the skeleton (its prior *is* a 1-ply lookahead); the fast
`_clone_battle` makes it cheap. Two sub-variants, in order:

1. **V-greedy turn planner**: pure play-time change, no training — reuse the best
   existing value head. If this alone jumps above 0.70, H2 is confirmed and the
   "reactive" deployment constraint should be renegotiated (it is barely slower
   than a forward pass per *turn*).
2. **Train the value for the planner** (fitted-value iteration against the
   planner's own play) if variant 1 shows signal.

Positioning: E4 asks "how much planning can be baked into weights"; E5 asks "what
is the minimum play-time compute that recovers most of search." They bracket the
gap from both sides, and both reuse the same value-head work.

### E5a — cheap once a value head exists: learned-potential reward shaping (PBS with learned Φ)

The principled version of "reward board strength." Hand-crafted board-structure
rewards are a trap this repo already walked into: PBS with a board-advantage
potential *hurt monotonically* (`ppo-review.md` §8.2, 0.554 → 0.500) because a
"strong board" prior discourages the face-trades that win this tempo game — and
sharper variants (pay for width, pay for holding removal) would teach
over-extension and card-hoarding respectively. Instead, use a **frozen learned
value function as the potential**: `r' = r + γ·V(s') − V(s)` with Φ = the best
existing value head (or E4's distilled one). Still policy-invariant (Ng et al.
holds for any Φ), but Φ now encodes guard walls, width, tempo, and removal timing
*as they actually correlate with winning* — including the tempo-over-control truth
that broke the hand-crafted version. Densifies credit assignment onto every
atomic action, which is also a second-order attack on the E3a sequencing problem.
Requirements: freeze Φ (never let it move with the learner); A/B on the standard
+0.03 ruler. Cheap once E4/E5's value-head work exists; expect modest — it
accelerates learning more than it raises ceilings, so read it as an
efficiency-and-sharpness lever.

(If board-structure *awareness* rather than reward is the goal: auxiliary heads
predicting next-turn face damage / own-guard survival are the zero-policy-bias
alternative — see E6; and one-line structural scalars like "protected attack
total" can ride the next obs-variant run, expectations low given two null obs
A/Bs.)

### E6 — medium-high: attack H3 — belief features / auxiliary opponent-prediction loss

Give the net what determinization gives the search: (a) cheap engineered belief
features (opponent mana curve, cards drawn vs played, known deck composition from
draft mode — currently the agent doesn't even see what the opponent has *played*
historically); or (b) an auxiliary head predicting next-turn opponent damage /
hand composition, trained self-supervised from replays, sharing the extractor.
Falsifies-or-confirms H3 cheaply before committing to recurrence.

**E6a — opponent hand-card age (scoped sub-item).** Per-card "turns since drawn"
for the opponent's hand, exposed as aggregates (mean/max age, count held >2
turns) appended to the tactical scalars. Rationale: a card held while mana was
available is evidence of something situational — a belief feature in the H3
family. Two caveats gate it:

- **Fairness:** exact ages from `GameState` truth leak slightly (when the
  opponent plays a card, a public observer cannot tell which draw it consumed).
  Use aggregates and/or ages tracked from the public event stream only.
- **Expected null vs the current eval set:** the hard3 baselines are greedy
  heuristics that dump their hand every turn, so age is nearly a deterministic
  function of `(turn, op_hand_count)` — both already observed. The feature only
  carries signal against opponents that sandbag (search policies, self-play).

**Falsification probe first (one script, no training):** log opponent hand ages
over a few hundred games vs each hard3 baseline and measure their variance
conditional on `(turn, op_hand_count)`. If near-zero (likely), skip the
training A/B entirely; revisit only if the training/eval opponent pool ever
includes card-holding policies.

### E7 — high: scale AZ Phase 2 (self-play of the search)

Known-null at 3 iterations x 140 games; the standard cure is 10–50x more self-play
games per iteration. The gain lands on the *search* policy (and via E4 flows back
into the reactive one). Only worth it after E4/E5 establish where the amortization
ceiling is — if E4 stalls at ~0.70 for information (H3) reasons, a stronger teacher
won't move the student.

### E8 — high: recurrent masked PPO

The full H3 treatment: history-conditioned policy (sb3-contrib has `RecurrentPPO`
but no *maskable* recurrent variant — custom work). Heavy engineering, uncertain
payoff, and E6 tests the same hypothesis at a fraction of the cost. Do not start
here.

### Explicitly not worth re-running

More HP search (R2/R4 verdict), bigger/deeper extractors (hurt or neutral),
*hand-crafted* reward shaping (ruled out, PBS-correct — the learned-Φ variant in
E5a is a different object and is not covered by that null), more self-play rounds
of the raw net (plateaus ~0.64), obs richness of the "more flat scalars" kind
(null twice).

---

## 3. Code inspection — mistakes found

Ranked by expected impact. 1–2 are real bugs affecting current conclusions; 3–5
are latent/robustness; 6–8 are minor.

### 1. Dropout is active inside the PPO importance ratio (likely root cause of the token net's KL blowup)

`TokenSetExtractor` defaults `dropout=0.1` (`extractor.py:58`). SB3 collects
rollouts in eval mode (`collect_rollouts` calls `set_training_mode(False)` — no
dropout) but computes `evaluate_actions` during `train()` in train mode (dropout
*on*). So the ratio `pi_new/pi_old` differs from 1 even *before any gradient
step*, purely from dropout noise: `approx_kl` and `clip_fraction` are inflated,
and every minibatch gradient carries ratio noise that the clipping then converts
into bias. This is a known PPO footgun ("don't use dropout in PPO"), and it
matches the documented symptom precisely: token net `approx_kl` 0.10–0.15 /
clip-fraction ~0.4 at default LR, "degrades with training", fixed by throttling
(`lr=1e-4` + `target_kl=0.025`). The tuned B0 recipe is plausibly compensating
for this. **Fix: `dropout=0.0` for PPO training** (E1). Note the R2 sweep tuned
*around* this noise source, which further muddies its (already null) result.

**Empirical resolution (2026-07-02): tested, and the fix was WRONG in net
effect — reverted.** The mechanism above is real, but removing dropout at the
tuned recipe regressed -0.028 paired, and the high-LR collapse happens with
dropout off too (N1: -0.247), so dropout was never the cause of the KL blowup —
it is a net-positive regularizer here. Kept at 0.1 with a pinning test; see the
E1 outcome note and the 2026-07-02 worklog N-battery entry.

### 2. token-v1 `op_reachable` / `exposed_to_lethal` computed from stale opponent readiness — undercounts threat

`encode.py:301-307` filters opponent creatures by `c.can_attack and not
c.has_attacked`. But `start_turn` refreshes *every* creature (`battle.py:64-66`),
so at the agent's decision point **all** opponent board creatures will be ready on
the opponent's next turn — and the filter excludes exactly the ones that attacked
last turn, i.e. the active attackers. Consequence: `op_reachable` systematically
understates incoming damage and `exposed_to_lethal` is 0 in many genuinely lethal
positions. The two headline scalars of the v1 variant are confidently wrong; run
R5 on this and a null verdict is uninterpretable. **Fix: sum attack over the whole
opponent board (keep the my-guard gate); drop or reinterpret the readiness filter.**
(The per-token `ready` bit for op-board tokens, `encode.py:192`, has the same stale
semantics — documented as low-signal for the flat path but silently inherited by
the token path.)

### 3. `WinRateEvalCallback` eval can silently never fire

`_on_step` gates on `self.num_timesteps % self.eval_freq != 0`
(`eval_callback.py:76`), but `num_timesteps` advances in increments of `n_envs`,
so the modulus hits only if `eval_freq` is a multiple of `n_envs` (it fires at
`lcm(n_envs, eval_freq)` intervals otherwise — possibly never within a run). The
sweep's `eval_freq` derivation comments this constraint but nothing enforces it,
and a future `n_envs=12` run gets one backstop eval and no pruning signal. **Fix:
track the last-fired bucket (`num_timesteps // eval_freq`) and fire on change, or
gate on `n_calls`.**

### 4. Parallel envs train on overlapping episode-seed streams

`_build_env` seeds env `i` with `seed + i` (`training.py:62`) and each env draws
episode seeds `base + ep` (`battle_env.py:152`) — so worker `i`'s episode `k` and
worker `j`'s episode `k + (i - j)` replay the *same* game seed (same draft, same
decks, same draw order). With 16 workers the rollout buffer contains near-duplicate
episodes offset by one, cutting effective sample diversity. **Fix: stride per-env
seeds (e.g. `seed + i * 100_000`), keeping the block disjoint from the 1M+ eval
range.**

### 5. `BattleEnv` never resets/reseeds the opponent policy

`run_game` resets both policies per game for determinism (`engine.py:107-108`);
`BattleEnv.reset` does not touch `self.opponent`. Deterministic scripted opponents
don't care, but a stochastic or stateful opponent (`mixed`, `mcts`, a frozen PPO
snapshot with `deterministic=False`) makes episodes depend on worker history —
irreproducible and subtly different across `n_envs` layouts. **Fix: call
`self.opponent.reset(eff)` in `reset()`.**

### 6. Behavior-cloned models have an untrained (random) value head

`behavior_clone`'s loss is `-log_prob` only (`distill.py:173,226`) — no gradient
reaches the value head. Harmless for pure play (argmax uses the policy head), but
dangerous in the two places a distilled model would plausibly go next: as a PPO
warm start (garbage V → destructive early advantage estimates) and as a `NetOracle`
for `netdmcts` (the value head drives leaf evaluation). E4's value-distillation
item fixes this; at minimum, document the hazard.

### 7. Simultaneous deaths resolve as a seat-1 win; draws count as agent losses

`check_winner` checks seat 0 first (`battle.py:100-104`), so if one action drops
both players to ≤0 (deck-out + summon self-damage, Drain edge cases), seat 1 wins
deterministically. And `BattleEnv.step` maps any non-agent winner to reward -1
(`battle_env.py:199`), while `run_match` credits `winner is None` to the B side —
consistent but silently biased. Rare enough not to move numbers; worth a comment
or an explicit draw convention.

### 8. Minor sweep nits

**Moot as of 2026-07-02: the Optuna sweep machinery (`locma/envs/sweep.py`, the
`sweep` CLI command, the `[sweep]` extra) was removed after the study concluded
with a null verdict — recover from git history if a re-sweep is ever needed.**
Preserved for the record: `objective` returned `-1.0` for infeasible configs —
TPE ingests it as a real (terrible) observation next to a hard boundary; `raise
optuna.TrialPruned()` keeps the surrogate cleaner. And `HyperbandPruner` with ~30
trials and 6 checkpoints has near-empty brackets (observed in R2: nothing pruned
until late) — with small trial counts `MedianPruner` behaves better.

---

## 4. Suggested sequence

1. **E1 + E2 + E3** (a day, mostly rerun time): fix dropout, threat scalars,
   eval gating; re-establish B0 with dropout off; run R5 on the *fixed* v1.
2. **E3a** (tactics suite): build the sequencing metric now, so every later
   experiment reports against it.
3. **E5 variant 1** (V-greedy turn planner, play-time only): the cheapest test of
   H2 — if it clears ~0.70 the planning hypothesis is confirmed and the
   deployment conversation changes.
4. **E4** (netdmcts distillation with soft targets + DAgger): the honest
   amortization ceiling; its value-distillation work feeds E5 variant 2, E5a's
   frozen Φ, and E7.
5. **E5a** (learned-potential shaping) once a trustworthy frozen value head
   exists — cheap A/B, read as an efficiency lever.
6. **E6** only if E4/E5 plateau in a way that implicates hidden information
   rather than planning; **E7/E8** only after that evidence exists.
