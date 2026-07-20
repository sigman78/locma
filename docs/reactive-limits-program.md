# Reactive-limits program — closing the net-vs-search gap from inside the net

Goal (2026-07-19): evolve the reactive PPO net itself — architecture and
training — toward the fair-search rungs. Play-time guards (E26 lens) stay in
production but are not the direction; no further guard work as an end.

Terminology (canonical, pinned to code): **action head** = `action_net`
(Linear 64→155; older entries say "policy head"), **value head** =
`value_net`, **extractor head** = `features_extractor.head` (the token
extractor's fuse layer), **towers** = the two 64x64 tanh MLPs
(`mlp_extractor.policy_net`/`value_net`), **extractor** =
`features_extractor`, **trunk** = extractor + towers (everything before the
two output heads; NB the policies code separately uses "trunk forward" for
one full net pass), **arm** = an experiment variant, never a network part.

## Where the gap is (established)

- **Perception is complete.** Everything needed is linearly decodable from
  the raw observation (E27: lethal 0.986, threat 0.977, can_item 0.983) and
  the trunk has no capacity pressure (net-probe prestudy, PR #83).
- **Routing defect.** The trunk destroys hand-item information on the way to
  the head (can_item 0.98 raw -> 0.61 towers) — mechanism for the 3.3x item
  underuse (E27).
- **Missing computation.** A LOCM move is a multi-action turn plan; search
  enumerates plans and argmaxes the critic over end-of-turn states. The net's
  single forward pass never performs that maximization (E14a: turn-level
  branching; E22: depth wins; E15: imitation "starts planner lines, can't
  finish them").
- The E15 "training-side program closed" verdict covered data-side levers on
  the EXISTING architecture (distill/imitation into the standard head,
  ranking loss on a FROZEN extractor, diversity, recurrence, obs encodings)
  — measured on a trunk we now know is ill-conditioned (94% first-layer tanh
  saturation) and info-destroying. It did not test changing the function
  class. That is this program.

## Arms (gates before benches; cheapest falsification first)

### E28 — pointer-style action head

Replace the dense 64x155 logit map: each Summon/Use logit is computed by
attention from that action's own hand-card token, each Attack logit from the
corresponding board token (pointer-network head over TokenSetExtractor's
per-slot outputs). Preserves per-card information by construction — the
direct fix for the E27 routing defect.

- Gate 1 (CPU-minutes): BC a pointer-head net on an existing practicum;
  run the E27 concept battery — can_item retention at the head must jump
  from ~0.61 toward ~0.95, and BC agreement must be >= the standard head's.
- Gate 2: PPO retrain at the b0k recipe; paired ruler vs ppo:b0k,ldraft with
  the boardkeep guard-rail (E26 protocol). Success = CI-positive; stretch =
  approach the lens-guarded 0.85 with no guard.

**VERDICT (2026-07-19): PROMOTED on both rungs** (worklog E28 gates 1-2 +
stack; baseline.md 2026-07-19 section; artifact `depot:e28p` v1). Gate 1: BC
agreement 0.423/0.436 vs 0.390-0.392 capacity-matched control — first break
of the ~0.37 imitation cap; but NOT via items (recall a wash — access was
never the item problem, refining the E27 mechanism). Gate 2 + stack: pure
net +0.073 headroom (0.856/0.858, milestone 1 cleared with no guard);
`lppo:e28p trio` 0.908 beats the guarded RoR (+0.059, confirmed); boardkeep
neutralized to 0.221. Items unmoved by slot access even under PPO — three
instruments agree it is consequence valuation (E30's question).

**E28b addendum (2026-07-19, worklog E28b): the gather does not need the
transformer.** Gate-1 protocol, one variable: a pointer head gathering
pre-attention slot embeddings — or even raw unprojected 33-d features —
matches pointer-over-z on both seeds (premix within 0.005, pre-registered
MIXING_UNNECESSARY). The BC-cap break is pure structural slot access.
Scope: the context path (latent_pi) still used the full trunk; a
transformer-free TRUNK is a retrain question, folded into E29 below.

### E28c — feature completion: play-effect columns (PREREQUISITE for E29)

Census (2026-07-19): **44/160 cards — including 7 of 8 blue items — have
play effects (player_hp / enemy_hp / card_draw) that are NOT in the 17
numeric token features.** The only path to that information is the card-id
embedding, which measurably never trains (encoder-viz: 0.997 cross-net
correlation with init, probes at chance). The net is blind to burn/heal/
draw semantics except via cost/type/stat correlations — a confound inside
the item-underuse mechanism (E27 probed can_item ACCESS, not effect
KNOWLEDGE) and an unfair handicap for any slim-extractor comparison.

Fix: opt-in token variant `fx` (`obs_mode="token-fx"`, TOKEN_FEATS_FX=20)
appends the 3 effect columns for HAND cards (board slots zero — effects are
spent on play); scalars stay v0; extractor reads token width from the obs
space; play-time consumers detect the variant via
`encode.token_variant_for_space` (scalar width alone can no longer
disambiguate). Default paths byte-identical.

- Gate (CPU-minutes): E28b-protocol BC — raw gather vs raw+fx gather.
  Asymmetric read: a positive on item behavior fast-tracks; a null does NOT
  kill (BC against a search teacher cannot price consequence value — the
  known residual). The real test is PPO.
- Bench: pointer-head retrain at the e28p recipe with token-fx, paired
  ruler vs the e28p RoR pair + boardkeep guard-rail (gate-2 protocol).
  E29 arms then build on whichever obs variant wins.

**VERDICT (2026-07-19, worklog E28c): CI-POSITIVE, replicated; items move
for the first time.** BC gate null-by-construction (the practicum has ZERO
blue items in hand — the training/deploy item-distribution gap is itself a
finding). PPO at the exact recipe: **+0.0212 [+0.0140, +0.0282]** vs the
e28p pair, confirm +0.0215 [+0.0157, +0.0273] — sub-headroom but
zero-excluding twice; item rate per opportunity **0.142 -> 0.19-0.23**, the
first instrument to move item behavior in the program; boardkeep stays
closed (0.28-0.30). E29 arms build on token-fx.

**PROMOTED on both rungs (2026-07-19, worklog E28c stack; artifact
`depot:e28c` v1).** 3-seed ladder (scripts/e28c_stack_bench.py), every CI
zero-excluding: pure trio +0.0170 [+0.0108, +0.0233] vs the e28p trio;
`lppo:e28c trio` **0.914** beats the 0.908 guarded RoR +0.0124 [+0.0089,
+0.0159], fresh confirm +0.0128 [+0.0079, +0.0177]; lens increment on fx
nets +0.0352 (headroom — still disjoint); boardkeep 0.2185 vs the stack,
matching e28p's 0.221. Sub-headroom promotion on the E7 precedent
(zero-excluding CI + fresh replication, three times over). New records:
reactive 0.878, guarded 0.914. Open follow-up: fx + item-rich training
decks (draft_override) as the motivated escalation.

### E29 — conditioned trunk (LayerNorm / input normalization)

Fixes the measured pathology (first-layer saturation with LOW PR), likely a
multiplier rather than a standalone lever: every historical null trained on
the handicapped trunk.

- Gate (probe-based, early checkpoints): saturation must drop AND can_item /
  control-concept retention must improve vs b0k at matched steps; else kill.
- If gates pass: rerun the ONE closed lever with the clearest unresolved
  mechanism — full-net ranking loss (E15's wall was explicitly
  frozen-extractor; E13 concluded the critic ceiling needs ranking-type
  signal). Target: single-net critic past the 0.890 single-critic ceiling.
- Added after E28b (mixing unnecessary for the pointer gather) + the
  encoder introspection (attention near-uniform, id embedding untrained):
  a **slim/transformer-free extractor arm** — per-slot embedding + pointer
  head + cheap context (scalars + pooled slots), retrained at the e28p
  recipe. Watch the critic: the vf tower is the one place the trunk
  genuinely computes (winner_side). Prize: the extractor was ~4x the flat
  net's compute; the reactive rung sells on cheapness.

### E30 — autoregressive turn-plan head (BC diagnostic first)

The ~0.37 BC-agreement cap (2026-06-27) was measured under per-micro-action
factorization; "starts lines, can't finish them" is that factorization
failing. Predict the turn's action sequence autoregressively (conditioned on
the plan so far), teacher-forced from planner/search practicums (labeler
machinery from E27 records everything needed).

- Gate: plan-level BC on held-out practicums. If sequence-level agreement
  clears the 0.37 cap decisively, open a training arm (EXIT-style with the
  plan head); if not, the cap is representational and this closes cleanly.

## Explicitly out of scope

Wider/deeper trunks (no capacity pressure, twice confirmed); more diversity
levers (one saturable resource, E7b/E7c/E13); recurrence (E6); auxiliary
concept-prediction losses (E27 finding 1: representing != using); further
play-time guards as ends (E26 stands as-is).

## Milestones

1. Pure net > 0.85 (lens-guarded RoR) — the net has internalized the lethal
   readout + item routing that guards currently patch. **REACHED 2026-07-19**
   (E28 pointer trio 0.865, no guards).
2. Pure net ~ 0.890 (single-critic vbeam ceiling) — the net has absorbed
   what one beam ply adds. 0.025 away after E28; E29/E30 are the remaining
   levers.
3. Stretch: approach 0.926 (ensemble planner RoR). (The GUARDED stack is
   already 0.908.)
