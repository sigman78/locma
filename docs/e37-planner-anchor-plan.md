# E37 — planner anchors in the PFSP pool: is the parity plateau REGIME or STRUCTURAL?

Pre-registered 2026-07-22 (launched same night). Branch/tag: `e37p` / `e37c`
chains via `scripts/e36_pfsp.py --tag`.

## Question

E36 closed the Phase-3 search wall to PARITY (3-seed pooled 0.509 [.481,.537]
vs `rbeam:shared`) and plateaued there — s22 sat at parity from gen4 through
gen7. Two readings:

- **Regime ceiling**: the pool composition is the limit. Every pool member
  (past selves + 4 scripted reactives) is REACTIVE — nothing in the regime
  punishes plan-level slop (a left-open lethal, an unconverted two-turn win),
  so best-response pressure never points at the planning tail. The E36
  autopsy is consistent: gains were trading/calibration; **missed-lethal
  stayed flat (~0.08)**, exactly the tail no pool member exploits.
- **Structural ceiling**: the reactive function class cannot compute
  plan-choice from the obs (E30's representational verdict; E22's "depth is
  the driver"). Parity is where training-side pressure runs out regardless
  of who is in the pool.

Discriminator: add a FAIR multi-turn planner as a pool anchor and
best-respond to it. If the regime was the limit, the held-out search gap
breaks BELOW parity and/or the missed-lethal tail finally moves. If
structural, the net may learn to beat the pooled planner narrowly (or not at
all) while the held-out gap stays at parity.

## Treatment

Continue the s22 chain (freshest parity endpoint on this box, n_envs=12
regime) 2 generations in TWO arms differing ONLY in pool:

- **e37p (planner)**: s22 terminal pool + `dmcts:15,60,0,3,ldraft` anchor
  (kind=anchor), initial weight 0.55 = the PFSP formula value from the known
  matchup (gen7 WR 0.448 vs it, dmcts ladder 2026-07-22) → ~16% of games.
- **e37c (control)**: s22 terminal pool byte-identical. Controls for scale —
  the x86 chain was still creeping down at gen7 on scale alone, so "2 more
  gens moved the gate" is uninterpretable without this arm (E11's budget-
  confound lesson).

Both: warm from `depot:e36s22/e36_s22_gen7.zip`, `--start-gen 8
--generations 2 --steps 1000000 --n-envs 12 --seed 24000000` (fresh base,
clear of 14/20/22M train and 5/30/40/60/70-72M eval bases). 1M steps/gen
(not 1.5M) on both arms — matched budget; the cut is the cost compromise
for the planner arm's 3.6x throughput tax (measured: control 297 fps,
planner 83 fps at n12 — dmcts search in-worker dominates).

Why `dmcts:15,60` and not `netdmcts`: (1) near-parity strength vs gen7 is
MEASURED (0.552 over the net) — the PFSP sweet spot; a much stronger anchor
gets high weight but a sparse win signal, a weaker one gets down-weighted
into irrelevance. (2) No net at the leaves → pure-CPU, ~6-25x cheaper than
`netdmcts:8,40` (~1s+/move; would triple gen time even at 10% sampling).
(3) E32: net-guided search quality does NOT track reactive-net strength, so
netdmcts's right oracle config is itself an open question. netdmcts stays
the escalation rung if dmcts moves nothing but internal WR rises.

## Ruler hygiene

- `rbeam:shared` is NOT in the pool — it stays the held-out primary gate
  (`scripts/e36_gate_gen4.py`, seed 30M, n=400/net, e29slim repro anchor).
- `dmcts:15,150` (hard rung) becomes the SECONDARY sensitivity ruler: gen7
  baseline 0.390 net-WR leaves headroom that the saturated 0.51-parity
  primary lacks. Caveat: same family as the pooled anchor (budget held out,
  family not) — read it as sensitivity, not proof.
- Behavioral tail: `scripts/e33_reactive_vs_search_behavior.py`
  missed-lethal rate (lguard ground truth). Baselines: e29slim 0.089,
  gen7 0.082 (flat). This is the instrument the user-facing hypothesis
  ("best-responding to planners teaches lethal conversion") lives or dies
  on.
- Internal: the driver's per-gen `wr_vs_pool` logs WR vs the dmcts member —
  the "can it best-respond to a planner AT ALL" discriminator, free.

## Pre-registered reads

| outcome | read |
|---|---|
| e37p gate < 0.50 (CI excludes .50), e37c ~ parity | **REGIME** — pool composition was the ceiling; escalate (stronger/more planners, netdmcts rung, arch-retest under richer signal) |
| both arms flat at parity, e37p internal WR vs dmcts RISES | **STRUCTURAL** (consistent with E30) — net exploits the pooled planner without generalizing; training-side program closes at parity, play-time search remains the answer |
| both flat AND internal WR vs dmcts flat | **signal starvation, NOT structure** — the planner wins too uniformly to teach; retry weaker rung (`dmcts:15,20`) or curriculum before concluding |
| e37p moves but exploit guardrail regresses | **E21 trap realized** — planner coverage traded against archetype coverage; adjust weights, never swap anchors |
| gate moves but missed-lethal flat | regime gain is real but the tail is (again) trading/pressure, not lethals — the lethal readout is a separate defect (E27's readout-failure line) |

Guardrail: `scripts/e36_exploit.py` on the e37p endpoint if promoted-track;
boardkeep must stay ≤ the 0.14-0.17 band (E21's regression signature is the
kill signal).

## E21 trap, explicitly

E21 (2026-07-08) swapped boardkeep out for a "stronger" net opponent:
reactive null, boardkeep guard REGRESSED, and the read generalized to
"opponent diversity from any source saturates; a different signal KIND is
needed." Defenses here: (1) ADD, never swap — all four scripted anchors keep
their weight floors; (2) matched-budget control arm; (3) a planner IS the
different signal kind E21's read called for (punishes plan-level mistakes
reactive opponents can't see) — and E36 already falsified E21's
generalization once (regime levers work where zoo swaps didn't); (4) ops:
dmcts is CPU-heavy in-worker — arms run SEQUENTIALLY (no box
oversubscription), throughput measured before launch, caffeinate wrapped.

## Cost (measured before launch, M1 Max n12)

| segment | fps | wall |
|---|---|---|
| e37p gen (1M steps) | 83 | ~3.3 h ×2 |
| e37c gen (1M steps) | 297 | ~56 min ×2 |
| driver eval_vs incl. dmcts member | — | ~10-15 min/gen |
| gate (5 nets × 400 games, 8 workers) | — | ~30 min |

Total ~10 h (overnight). Artifacts: `runs/e36_e37{p,c}_gen{8,9}.zip`,
`runs/e36_e37{p,c}/{pool,history_gen8+}.json`, `runs/e36_e37/gate.json`.

## VERDICT (2026-07-23, worklog E37)

**Three-instrument NULL — the anchor was INERT at this dose.** Held-out gate
identical between arms (gen9 0.475 vs 0.472); no specific adaptation even vs
the pooled `dmcts:15,60` itself (0.545 vs 0.5275, n=400 CRN); missed-lethal
flat (0.090 vs 0.108). The internal WR climb vs dmcts was generic self-play
gain (control matched it unseen). Guardrails clean (boardkeep 0.82-0.855,
e29slim anchor 0.812 4th repro). Per the pre-registered table this is the
signal-starvation row strictly (dose escalation to ~50% share remains
untried), but combined with E30/E33/E36 the structural reading carries:
what search adds is not learnable by playing against it. Full entry:
docs/worklog/2026-07.md 2026-07-23.

## E37b addendum (2026-07-23) — lguard anchor: sharp, attributable lethal punishment

The dmcts null's mechanism read (diffuse multi-turn pressure -> credit
assignment fails) leaves one anchor class untested: `lppo:` — a
lguard-wrapped self that NEVER misses its own forced win. Its punishment is
temporally adjacent (leave a lethal open -> lose next turn, every time),
exactly the reward shape PPO can attribute. Distinct hypothesis from E37
proper; the E37 null does not cover it.

- **e37l**: s22 pool + `lppo:runs/e36_s22_gen7.zip,ldraft` anchor, weight
  0.51 = 1 − measured pure-vs-guarded H2H (0.487, n=400, seed 26M) → ~15%
  of games. Same warm/seed/budget as e37p/e37c — **the existing e37c arm is
  the control** (nothing rerun).
- Cost: NO tax — 367 fps measured (the DFS is negligible), ~45 min/gen.
- Calibration note: the guard flips only ~2.6% of mirror H2H games, so the
  punishment channel is sharp but SPARSE — a null here may mean "signal too
  rare at 15% share," not "unlearnable"; the missed-lethal autopsy
  disambiguates (defensive lethal-avoidance should move even if conversion
  doesn't).
- Instruments, same battery: held-out `rbeam:shared` gate (ladder
  `e37l_gen{8,9}`), e33 missed-lethal vs e37c's 0.108 / the 0.08-0.11 band,
  internal per-gen WR vs the lguard member (driver `wr_vs_pool`), boardkeep
  driver-side watch.

## E37b VERDICT (2026-07-23, worklog E37b)

**Null at high power.** Gate flat (0.495/0.492 vs control 0.497/0.472);
missed-lethal at n~1600 lethal-turns/arm: e37l 0.0870 [.074,.102] vs e37c
0.0960 [.082,.111] (z=0.80) — the 150-game read's suggestive 0.073 was
instrument noise (the standard autopsy is ±0.05 on this metric; use
`--games 250`, it costs ~a minute). Any true effect bounded <3pp. Internal
WR vs the anchor trendless; items/face/disagree in band. Three arms, three
signal shapes, every instrument identical: at parity, pool composition no
longer matters — the sparse-signal branch would need a ~50%-share lguard
arm (free, no tax) for the dose-response point, but the structural verdict
stands. Parity is the training-side endpoint; lethal conversion stays
play-time (E26).

## E37c addendum (2026-07-23, pre-registered before launch) — lguard dose-response: 30% and 50% share

E37b's null strictly licenses only "no effect at ~15% share x 2M steps"; its
own calibration caveat (guard flips ~2.6% of games — sharp but SPARSE) is the
open branch. This runs the dose-response point the E37b verdict named: same
anchor, 2x and 3.3x the exposure.

- **Arms** (sequential, 50 first — the max-contrast point): **e37l50** =
  s22 terminal pool + `lppo:runs/e36_s22_gen7.zip,ldraft` at **50% of
  games**; **e37l30** = same at **30%**. With e37c (0%) and e37l (~15%)
  this gives a 4-point dose curve on every instrument.
- **Dose control**: new `pin_share` pool-entry field in `e36_pfsp.py` — the
  PFSP reweight (`1-wr`, the thing that held e37l at ~15%) skips pinned
  members; after each admission the pinned weight is re-solved so the
  sampling share stays exactly at the dose. Verified offline (share 0.5000
  both gens through admit/evict). ADD-not-swap holds: all scripted anchors
  keep weight floors; their absolute share dilutes at 50% pin — that
  dilution is inherent to the dose lever, and the boardkeep watch + held-out
  gate are the E21-trap instruments for it.
- **Everything else byte-matched to e37l/e37c**: warm
  `depot:e36s22/e36_s22_gen7.zip`, `--start-gen 8 --generations 2 --steps
  1000000 --n-envs 12 --seed 24000000` (CRN with both prior arms). **NO
  `--deck-pool`** although it merged today (#105): the cached pool's 80/20
  ldraft/random mixture is a deck-distribution change and would confound
  dose with decks against the live-draft e37c control. The speedup applies
  to future chains where all arms share it.
- Cost: lppo is tax-free (367 fps measured, E37b) — ~45-55 min/gen,
  ~4-4.5 h both arms incl. driver evals; gate + autopsies ~40 min after.
- Instruments, same battery: held-out `rbeam:shared` gate (ladder
  `e37l{30,50}_gen{8,9}`, e29slim anchor repro alongside), e33 missed-lethal
  at `--games 250` (the E37b instrument lesson; n~1600 lethal-turns/arm),
  driver `wr_vs_pool` (internal WR vs the lguard member + boardkeep watch),
  items/face/disagree bands.

Pre-registered reads:

| outcome | read |
|---|---|
| missed-lethal falls monotonically with dose (50 < 30 < 15/0 band), gate holds | **sparse-signal branch was real** — the lethal tail IS trainable given enough adjacent punishment; dose curve locates the knee; promotion question opens |
| both doses flat (missed-lethal in the 0.08-0.11 band, gate at parity) | **structural verdict FINAL** — signal-starvation branch closed at 3.3x dose; pool composition is exhausted as a lever at any feasible share |
| gate regresses at 50% (and/or boardkeep watch drops below band) | **E21 trap at high dose** — anchor share traded against archetype coverage; the dose lever is bounded by coverage, not learning; read 30% arm as the usable ceiling |
| internal WR vs lguard member rises with dose but missed-lethal flat | narrow exploitation of the guarded mirror (E30-consistent) — the net learns the matchup, not the lethal concept |

## E37c VERDICT (2026-07-23, worklog E37c)

**Row 2 — flat at both doses; the structural verdict is FINAL.** Missed-
lethal dose curve 0/15/30/50%: 0.0955/0.0873/0.0953/**0.1047** (n~1600
lethal-turns each) — flat-to-adverse (50 vs 15: z=+1.7 the WRONG way; every
dose vs control: null). Gate at parity everywhere (l30 0.4575/0.505, l50
0.485/0.4425; e29slim 0.8125, 5th repro); l50_gen9's marginal sub-.50 read
is noise-consistent (non-monotone in dose, one-of-four multiplicity, 3pp
CRN delta vs control) and its own missed-lethal is the worst of the four.
Internal WR vs the guarded mirror trendless at every dose (no narrow
exploitation); boardkeep watch clean-to-better (0.84-0.885) — no E21 trap
from halving the scripted anchors' share. Dose held exactly (pin_share
driver logs: 0.5/0.3 both gens; no throughput tax, 382-468 fps). At 3.3x
exposure the sharpest attributable lethal punishment moved nothing:
**pool composition is exhausted at any feasible share; parity is the
training-side endpoint; lethal conversion stays play-time (E26).** The
E37 program (a/b/c — planner, lguard, dose) is closed.
