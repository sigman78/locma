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
reactive 0.878, guarded 0.914.

**Escalation CLOSED (2026-07-20, worklog E28d): fx + item-rich training
decks is CI-NEGATIVE.** After closing the training-data hole (E31a: correct
per-card values incl. hidden effects + values-JSON `--draft-override`; diet
decks 6.28 items / 1.13 blues vs ~0 before), retraining e28c's exact recipe
on them regressed the ruler -0.0260 [-0.0343, -0.0181]. The item-conversion
gain was already saturated by fx alone at the item-light diet: e28c plays
blues at 0.202/opportunity on blue-rich decks, e28d (drenched in blues) at
0.19/0.14 — not higher; deploy-deck item rate unchanged; general play lost
to distribution shift. The E31a table and plumbing are correctness keepers
(E31 non-goal: any draft-side strength claim).

**Refinement (blue-value diagnostic, scripts/blue_value_diag.py): the E28d
null hides a real capability gain, and item underuse on today's blues is
mostly CORRECT.** A cheating perfect-information oracle plays blues at only
0.220/opportunity (items 0.302) — an optimal informed player declines ~78%
of blue chances, so blues are contextually weak (e28c 0.170 is a modest
underuse gap; e28d 0.243 is at/above oracle). And a magnitude-dose probe
(scale a blue's fx effect columns, read the fx net's play prob) shows a
dissociation: e28c is FLAT (plays blues ~0.20 regardless of strength —
magnitude-blind), e28d is MONOTONE (+0.05 over k0->k3 — plays a card more
as its effect grows). Item-rich training bought magnitude-conditioned
valuation that did not pay off on weak cards but WOULD scale with stronger
item design — retain e28d as that artifact. So the item residual is two
parts: (a) a small consequence-valuation gap on strong lines (E30), and
(b) card design — today's blues are too weak for even an oracle to play
often. Exposure helped the representation, not win rate.

**Oracle caveat (scripts/blue_oracle_horizon.py): "blues weak" splits by
effect type — the oracle undervalues card-draw blues.** The oracle's leaf
value (health + board-power lead) has no card-advantage term, so it is
blind by construction to the 2 draw blues (154, 157). A rollout-horizon
sweep confirms it: draw-blue play rate rises as the rollout extends to
terminal (real win/loss) while non-draw stays flat (~0.18-0.21, already
priced). Verified at 150 games / fresh seed with bootstrap CIs
(scripts/blue_oracle_horizon_verify.py): terminal - base draw-blue =
**+0.083 [0.021, 0.143]** (excludes zero); draw-blues go from below
non-draw at base (0.163 vs 0.218) to parity/above at terminal (0.246 vs
0.210). So (b) holds for the 6 removal/heal/burn blues but is OVERTURNED
for card-draw blues — and the fx net playing draw-blues at ~0.30 (which it
can see via the card_draw column) looks correct, not a mispricing. Don't
treat the cheating-MCTS oracle as ground truth for draw/tempo items.
(Terminal rollout is still random play, so true draw value may be higher.)

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

**E29a VERDICT (2026-07-20, worklog "E29a"): CLOSED (negative).** The
`feature_ln` lever (LayerNorm on the tower input; opt-in, default-off,
`train-zoo --feature-ln`) HALVES first-layer saturation (pi_a1 0.566 ->
0.258) — gate 1 passes — but can_item tower-retention does NOT move (0.745
-> 0.741; gate 2 fails). So saturation is DECOUPLED from the item info-loss
it was hypothesized to cause. A tiebreaker paired ruler (proxy-independent)
is CI-NEGATIVE: conditioned -0.0147 [-0.0245, -0.0053] vs matched control
(seed 0, 500k). First-layer saturation is apparently a COSMETIC pathology,
not the training handicap the prestudy framing assumed — which weakens the
"ill-conditioned trunk gates the historical nulls" rationale and the
ranking-loss-rerun plan below it. The slim/transformer-free extractor arm
(next bullet) is a SEPARATE hypothesis (cheapness, not conditioning).

**Slim-extractor VERDICT (2026-07-20, worklog "E29 slim extractor"):
PROMOTION CANDIDATE — it BEATS e28c, not just matches.** SlimTokenExtractor
(transformer dropped; per-slot embeddings + pooled context; 56.7k extractor
params vs 418.5k, 7.4x fewer) at the exact e28c recipe: paired ruler
+0.0259 [+0.0198, +0.0320] @ 58M, confirm +0.0282 [+0.0208, +0.0354] @ 59M
(0.903 vs 0.877), boardkeep in-band. Removing the transformer improved win
rate — consistent with E28b (mixing unnecessary) + the encoder-viz nulls
(near-uniform attention, untrained id embedding): the transformer added
overfitting capacity, not signal. s2 + stack ladder (vs the 0.914 e28c
guarded RoR) decides promotion, mirroring the e28c flow.
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

**VERDICT (2026-07-20, worklog "E30"): CLOSED — the cap is REPRESENTATIONAL.**
Controlled BC diagnostic (scripts/e30_plan_bc.py, mcts:100 practicum, turns
split by round counter): a factored head reproduces the cap (multi-action
agreement 0.373) and an autoregressive head that also sees the plan so far
does NOT clear it (0.366; autoreg-factored = -0.008/-0.005/-0.004 across
seeds 0/1/2, vs the +0.10 open-arm bar). Explicit plan-so-far conditioning is
redundant with the state (the board already reflects actions taken this
turn), so factorization is NOT the bottleneck. The reactive obs does not
separably encode which action a lookahead teacher picks — play-time search
fills that. No training arm opened.

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
