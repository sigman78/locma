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
