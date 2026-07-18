# Worklog — policies & engine

Brief running log of experiments and insights. Newest entries at the bottom of
each day. Detail lives in `docs/ppo-review.md` and `docs/baseline.md`; this is the
scannable index. One line per finding.

---

Entries live in `docs/worklog/`, split by month.

## 2026-07

- 2026-07-18 — [Net-probe instrument: NN-utilization metrics before the arch sweep](worklog/2026-07.md) — PR/erank/saturation/CKA/linear-probe instrument shipped; pilot: no capacity pressure in either net, no layer beats raw obs on teacher-action decoding — conditioning and head, not width, are the sweep candidates
- 2026-07-09 — [E22: MCTS-depth pilot vs the planner RoR](worklog/2026-07.md) — both cheating MCTS and fair dMCTS cross over and beat the 0.978 planner at a shallow budget (~100-1500 sims); depth, not information, is the driver — vbeam only searches one turn deep
- 2026-07-08 — [E21: advanced-opponent zoo pilot](worklog/2026-07.md) — swap boardkeep for an old-net+edraft opponent: reactive null, critic +0.013 (likely saturated), boardkeep guard REGRESSES — lever closed
- 2026-07-07 — [E20: draft-priority distillation](worklog/2026-07.md) — per-card fit vs real picks is null (context confound); census heuristic (+0.067) and net-elicited priority (+0.107 reactive/+0.060 planner) both recover real headroom; depot:edraft published as a usable draft policy
- 2026-07-07 — [E19: deck-distribution retrain](worklog/2026-07.md) — reactive NULL (pilot is deck-robust), critic +0.010 CI-positive small; shared-recipe retrain is the surviving lever
- 2026-07-07 — [E18c: exploit re-read of the ldraft pair](worklog/2026-07.md) — guard-rail pass; every archetype worse by 0.11-0.15, boardkeep neutralized (0.51 to 0.38-0.41)
- 2026-07-07 — [E18b addendum: PROMOTION — depot:ldraft](worklog/2026-07.md) — both recipes of record swap their draft half; planner 0.978, reactive 0.791
- 2026-07-07 — [E18b: learned draft beats the balanced draft — BOTH arms confirmed with headroom](worklog/2026-07.md) — headroom on both rungs; learned draft beats the scripted balanced draft
- 2026-07-06 — [E18a: gamma=1.0 retrain (ByteRL ablation transfer test)](worklog/2026-07.md) — CI-negative; ByteRL's gamma=1.0 ablation does not transfer to on-policy PPO
- 2026-07-06 — [E17: draft item-discount sweep under the planner](worklog/2026-07.md) — null-to-negative, monotone dose; draft-side spell enrichment closed
- 2026-07-06 — [E16a: spell-representation diagnostic (observability hypothesis)](worklog/2026-07.md) — null; observability hypothesis dead, underuse uniform regardless of visibility
- 2026-07-06 — [E15: ranking-loss critic + AZ-v2 policy round -- BOTH STAGES RESOLVE NEGATIVE, with mechanism](worklog/2026-07.md) — both stages negative; frozen-extractor capacity wall, training-side program closed
- 2026-07-06 — [E14a: within-turn failure diagnostic -- the obstacle is branching, not memory](worklog/2026-07.md) — intra-turn-context hypothesis refuted; failure is turn-level branching
- 2026-07-06 — [E13: boardkeep zoo + shared draft -- NO STACK; the diversity lever is one saturable resource](worklog/2026-07.md) — no stack; confirms the training-diversity critic lever is one saturable resource
- 2026-07-05 — [E12: b0k promoted to reactive recipe of record; mixed ensembles -- saturated at 3 critics](worklog/2026-07.md) — b0k promoted to reactive recipe of record; cross-family ensembling saturates
- 2026-07-05 — [E11: boardkeep into the training zoo -- hardening is PARTIAL, but the critic likes the data](worklog/2026-07.md) — partial hardening (parity not neutralization); critic gains, no promotion
- 2026-07-05 — [E10: adversarial exploit benchmark -- the reactive net IS exploitable; the planner is not](worklog/2026-07.md) — reactive net loses to a scripted exploit (boardkeep); planner rungs hold
- 2026-07-05 — [E9: distill the critic ensemble into one checkpoint -- VERDICT: null (values transfer, orderings do not)](worklog/2026-07.md) — null; ensemble values distill fine, sibling orderings do not transfer
- 2026-07-05 — [E8 addendum: fresh-anchor confirm + PROMOTION](worklog/2026-07.md) — promoted; 3-critic ensemble is the new planner recipe of record (0.926)
- 2026-07-04 — [E8: zero-training trio -- critic ensemble clears the bar; width curve pins the bottleneck](worklog/2026-07.md) — headroom, +0.036, zero training cost; 3-critic ensemble is evaluator-limited fix
- 2026-07-04 — [E7c: shared+rnd4 stacking -- no additivity; mechanism fully pinned](worklog/2026-07.md) — no additivity; diversity lever fully pinned as one saturable resource
- 2026-07-04 — [E7b: rndK dose-response -- mirror-breaking saturates fast; shared stays the best point](worklog/2026-07.md) — saturates fast at low noise dose; shared draft remains the best point
- 2026-07-03 — [E7 addendum 2: mechanism discriminator -- mirror-breaking itself is most of the effect](worklog/2026-07.md) — mechanism pinned: generic mirror-breaking diversity, not draft structure
- 2026-07-03 — [E7 addendum: fresh-seed confirm + PROMOTION](worklog/2026-07.md) — promoted; depot:shared is the new planner recipe of record (0.890)
- 2026-07-03 — [E7: shared draft variant -- first positive training-side CI, via the critic under vbeam](worklog/2026-07.md) — first CI-positive training-side lever (+0.026), via the planner's critic
- 2026-07-03 — [E4v2: distill the vbeam teacher + EXIT round -- VERDICT: null (plans do not compress into a reactive policy)](worklog/2026-07.md) — null; beam plans do not compress into a reactive policy via imitation
- 2026-07-02 — [Artifact depot: checkpoints of record move to `depot:` refs](worklog/2026-07.md) — versioned, provenance-tracked artifact depot shipped (PR #58)
- 2026-07-02 — [E5 variant 2b: AZ-style backed-up targets -- VERDICT: null (critic is already a fixed point)](worklog/2026-07.md) — null; critic is already a fixed point of its own one-step backup
- 2026-07-02 — [E5 variant 2: fitted-value iteration on the vbeam critic -- VERDICT: null-to-negative](worklog/2026-07.md) — null-to-negative; MC value targets erase the sibling ordering the beam needs
- 2026-07-02 — [E5 "planning-lite": vbeam own-turn beam planner -- H2 CONFIRMED](worklog/2026-07.md) — headroom, +0.206; play-time beam search beats the reactive plateau
- 2026-07-02 — [Draft-noise study: partial random draft](worklog/2026-07.md) — null; training-side deck diversity buys nothing for the reactive net
- 2026-07-02 — [Post-study cleanup: sweep + puffer machinery removed](worklog/2026-07.md) — study-only tooling removed now both ceiling-study verdicts are in
- 2026-07-02 — [R5-proper: token-v1 (fixed) at the winning recipe -- VERDICT #2](worklog/2026-07.md) — null; obs-encoding lever closed, B0 stays the reactive recipe of record
- 2026-07-02 — [N-battery results + dropout revert](worklog/2026-07.md) — dropout was a feature, reverted; ceiling verdict hardened
- 2026-07-02 — [E1-E3 fixes + benchmark](worklog/2026-07.md) — dropout removal regressed the net; E2/E3 correctness fixes landed

## 2026-06

- 2026-06-30 — [PPO ceiling study: Gate 0 (throughput) + kickoff on the RTX 4080 box](worklog/2026-06.md) — VERDICT #1: HP-tuned candidate confirmed regression vs B0, ceiling confirmed
- 2026-06-30 — [`locma-replay/3`: compact on-disk replay format](worklog/2026-06.md) — ~75% smaller, byte-for-byte lossless replay format shipped
- 2026-06-28 — [draft benchmark (`draft-bench`): isolate deck-building skill; `balanced` is robustly best](worklog/2026-06.md) — balanced is the Condorcet-winner draft under every strong pilot
- 2026-06-27 — [netdmcts Phase 2 (AlphaZero self-play of the search): no gain at this budget](worklog/2026-06.md) — null; one AZ round neither helps nor hurts, frozen-oracle netdmcts stands
- 2026-06-27 — [netdmcts (AlphaZero-lite Phase 1): a FAIR policy that beats the cheaters](worklog/2026-06.md) — fair net-guided search (0.817) beats the cheating mcts/azlite (~0.74)
- 2026-06-27 — [Self-play of token PPO2: responds where the flat net decayed (then plateaus)](worklog/2026-06.md) — real but front-loaded gain, plateaus ~0.64, ceiling intact
- 2026-06-27 — [Distill search → PPO2 (PR #18 redo): the obs is NOT the BC ceiling](worklog/2026-06.md) — null; behavior-cloning a search policy caps ~0.37 agreement regardless of obs
- 2026-06-26 — [PPO2: tokenized observation + self-attention (richer-encoding lever)](worklog/2026-06.md) — token obs reaches parity with flat, within seed noise; secondary lever
- 2026-06-26 — [searchers cheat; search-as-training-opponent; curriculum endpoint](worklog/2026-06.md) — mcts/azlite are perfect-info cheaters; dmcts is the only fair searcher
- 2026-06-26 — [full-roster tournament + rating-estimator fix](worklog/2026-06.md) — azlite undefeated; order-dependent rating bug found and fixed
- 2026-06-25 — [PPO investigation & draft control](worklog/2026-06.md) — action-space fix unlocks PPO; deck is the lever; self-play/distillation ruled out
