# E38 — afterstate consequence columns: answering E30's representational verdict

Status: PRE-REGISTERED 2026-07-24, before any run. Program doc
`docs/reactive-limits-program.md`. Branch `feat/e38-conseq`.

## Question

E30 closed the turn-plan arm with a **representational** verdict: "the reactive
obs/features do not separably encode WHICH action a lookahead teacher picks."
E37 then closed pool composition at any dose, and its verdict named the same
thing structurally. Both are statements about **a fixed representation**.

The literal response to a representational verdict is to change the
representation — and the one precedent for that in this program WON: E28c
(`token-fx`) appended 3 play-effect columns and returned +0.0212 [+0.0140,
+0.0282] plus the first instrument ever to move item behavior. Obs changes are
NOT on the program's out-of-scope list; what E14a/E16a closed was *encodings of
information already present* (intra-turn context, spell re-representation), and
what E30 closed was *plan-so-far conditioning* (redundant with the board).

E38 asks the untried version: **what if the observation carries, per candidate
action, what that action actually does — computed by the engine rather than
inferred by the trunk?**

## Why this is not a closed lever

| closed arm | what it did | why E38 differs |
|---|---|---|
| E14a / E16a obs work | re-encoded information already in the obs | these are NEW facts (a simulated afterstate), not a re-encoding |
| E30 autoregressive head | conditioned on the plan SO FAR (the past) | conditions on the consequence AHEAD (the future) |
| E27 aux concept losses | asked the trunk to PREDICT a concept | hands the concept over as input; E27's own finding is "representing != using" |
| E34 board-potential shaping | Φ as a REWARD (symmetric potential → net went passive, CI-negative) | the same Φ as an OBSERVATION. Different channel, different failure mode |
| E26 `lguard`/lens | lethal solve as a play-time OVERRIDE of the net's move | the lethal flag as an INPUT the net can learn to use (or ignore) |

Three of the program's live residuals are exactly per-action quantities the net
must currently infer through a pair interaction it demonstrably does not
compute (E27: the trunk computes exactly ONE concept beyond its input):

- **trade value** — E33: reactive trades are 29% lower-dphi and 37%-relative
  more often outright bad than the plan's. `dphi` is column 1 below.
- **overkill** — E33: +15% wasted attack per trade, an attacker→target
  MISMATCH. The pointer head already sees attacker atk and target def in
  `z_src`/`z_tgt` and still mismatches; the *joined* value is column 6.
- **missed lethal** — flat at 0.082–0.105 through E28/E29/E36 and all four E37
  doses; the only thing that ever moved it is a play-time exact solve (E26).
  Column 8 is that solve's answer as a feature.

## Fairness (load-bearing — read before implementing)

`CONTEXT.md`: the forward model is a perfect-information view, so policies that
read it are "cheating". E38 must therefore be **explicitly restricted to
own-side information**, on exactly the argument that makes E26's `lguard` fair:

1. Only the agent's **own** candidate actions are simulated. The opponent never
   moves in the simulation, so their hidden hand/deck is never consulted.
2. Features read only **public or own-known** state of the afterstate: board
   contents/stats, both heroes' health, own mana. Never opponent hand/deck.
3. **Self-leak guard:** if an action draws cards for us, the drawn cards are NOT
   read. A player knows their deck's *contents* but not its shuffled *order*
   (the same reasoning behind `determinize(reshuffle_own=True)`), so reading a
   drawn card would leak our own future. Columns are restricted to board/health/
   mana/lethal quantities, and the recorder asserts hand contents are unread.
4. `exposes_lethal` (column 9) uses the opponent's **visible board** only — the
   identical computation the shipped `v1` scalar variant already performs
   (`op_reachable`, `exposed_to_lethal`).

**Labeling caveat (pre-committed):** this is a 1-ply engine peek, so an
`ac`-variant net is NOT purely reactive. It is a rung *between* reactive and
search, and must be reported with the same discipline E26 used for `lppo`
("guarded reactive"), never folded into a pure-reactive number. Cost stays
~1× net inference plus ~5 engine applies per decision (measured branching
factor 5.1, `baseline.md` 2026-07-09); the engine runs ~10k turns/s against the
net's 0.15 s/game, so the deployment cost class is unchanged. The gate reports
measured s/game regardless.

## The columns (K=10, per semantic action index, zeros for illegal)

Computed on a clone of the decision state with the one action applied.
Φ = Σ(atk+def) over my board − same over op board (E33/E34's exact potential).

| # | column | why it is in |
|---|---|---|
| 0 | `dphi` = Φ(after) − Φ(before) | E33: the quantity measured 29% too low |
| 1 | `dmg_face` = op_health before − after | tempo, the axis the net over-weights (face-share 0.50–0.59 vs plan 0.48) |
| 2 | `d_my_health` | heals / self-damage (red/blue item effects) |
| 3 | `n_kills` = op creatures removed | board control, the E33 headline |
| 4 | `n_losses` = my creatures removed | the bad half of a trade |
| 5 | `overkill` = max(0, atk − target def) for a creature-attack, else 0 | E33: +15%, a pair mismatch the net does not compute |
| 6 | `mana_after` | curve/sequencing within the turn |
| 7 | `wins_now` = 1.0 if this action ends the game in my favour | trivially decodable but currently a readout failure (E27) |
| 8 | `enables_lethal` = 1.0 if an exhaustive own-turn lethal exists AFTER this action | the 0.082–0.105 tail nothing training-side has moved |
| 9 | `exposes_lethal` = 1.0 if op's visible board can kill me next turn | the defensive mirror; makes "don't tap out" decodable |

Column 8 uses `lguard.find_lethal` behind a cheap necessary-condition filter
(total reachable own attack ≥ op health) so most states skip the DFS; the
recorder reports filter hit-rate and cost.

## ADDENDUM 1 (2026-07-24) — the ruler is RE-PINNED, and the crossover is located

`scripts/e36_dmcts_ladder.py` (generalized: `--nets/--rungs/--pool-nets`, all
three PFSP parity endpoints registered), n=400/cell, matched `ldraft` both sides,
CRN per rung, `runs/e38/repin_dmcts.json`. **Cross-box validation first: this M1
box reproduces the x86-measured cells exactly** — e29slim 0.205/0.155 (published
0.205/0.155), gen7 0.4475/0.3900 (published 0.448/0.390). The ladder is portable.

Net WR over the fair `dmcts` oracle (higher = the net beats search harder):

| rung | sims | e29slim | x86 gen7 | m1 gen7 | s22 gen7 | **3-seed pooled** (n=1200) |
|---|---|---|---|---|---|---|
| med | 900 | 0.205 [.168,.247] | 0.4475 | 0.5050 | 0.5075 | **0.4867 [.459,.515]** — parity |
| hard | 2250 | 0.155 [.123,.194] | 0.3900 | 0.3950 | 0.4350 | **0.4067 [.379,.435]** — behind |

**`dmcts:15,150` (2250 sims) is the new primary ruler.** Pooled 0.4067 with a
CI that EXCLUDES 0.50 — 0.093 of win rate below parity, headroom in both
directions, ±0.028 at n=1200. The med rung is *also* saturated (straddles 0.50),
so it does not qualify; use it as the secondary/sensitivity rung.

**New result, worth its own line: the crossover is located between 900 and 2250
sims.** gen7 is at parity with fair determinized search up to ~900 sims and
measurably behind at 2250. E22 put the *planner's* crossover at 450-1500 sims, so
E36 pushed the crossover roughly 2x deeper in search budget — it did not remove
it. This is the precise version of "E36 reached parity": true against
`rbeam:shared` 8,20,4,4 and up to ~900 dmcts sims, false beyond. Seed spread at
the hard rung is 0.045 (~2 SE), with s22 nominally strongest — the same ordering
it had on the rbeam gate (0.4975), mild evidence it is the genuinely best endpoint.

## ADDENDUM 2 (2026-07-24, same round) — the motivation above is PARTLY STALE

The E33 numbers cited in the motivation table were measured on **e29slim**
(worklog 2026-07-20). Re-measuring on gen7 with the newly added reply-aware
oracle (`scripts/e33_reactive_vs_search_behavior.py --oracle rbeam`; 360 games,
subject `ppo:runs/e36_gen7.zip,ldraft` = the **M1** endpoint `depot:e36m1`, NOT
the x86 RoR; both oracle arms run on the subject's own net, identical subject
trajectories so the two arms are exactly paired) shows **E36 has since closed two
of the three residuals E38 was justified on**:

| E33 metric | e29slim (2026-07-20) | gen7 vs vbeam plan | gen7 vs rbeam plan | status |
|---|---|---|---|---|
| trade dphi | 1.293 vs 1.818 (−29%) | 2.287 vs 2.411 (−5%) | 2.287 vs 2.283 (**parity**) | **CLOSED** |
| attack face-share | 0.589 vs 0.482 (+0.107) | 0.535 vs 0.543 | 0.535 vs 0.524 | **CLOSED** |
| overkill / trade | 0.918 vs 0.795 (+15%) | 0.798 vs 0.688 (+16%) | 0.798 vs 0.674 (**+18%**) | **OPEN** |
| item play rate | 0.181 vs 0.226 (−20%) | 0.177 vs 0.239 (−26%) | 0.177 vs 0.226 (−22%) | **OPEN** |
| trades/turn | 0.703 vs 0.910 (−23%) | 0.765 vs 0.814 (−6%) | 0.765 vs 0.860 (−11%) | partly open |
| actions/turn | — | 3.202 vs 3.422 | 3.202 vs 3.461 | open (plays less) |
| mana unspent | 0.423 vs 0.417 | 0.564 vs 0.493 | 0.564 vs 0.495 | mildly open |
| root disagree | 0.357 / 0.377 | 0.380 | **0.406** | see below |
| missed lethal | 0.102 (pure) | 0.045 | 0.045 (identical) | consistency check ✓ |

**Consequences for E38 — recorded, not silently patched:**

1. **Column 0 (`dphi`) is no longer justified by a gen7 trade-VALUE gap.** gen7
   trades at the reply-aware plan's own dphi (2.287 vs 2.283). It stays in the
   block as the substrate for trade *frequency* (still −6% to −11%) and because
   the column ablation will now actually test it rather than assume it.
2. **Column 5 (`overkill`) is the strongest surviving motivation** — the ONE E33
   metric that did not move at all, and it gets *worse* against the deeper
   oracle (+18% vs rbeam). It is exactly a `(src, tgt)` pair quantity the
   pointer head sees the ingredients of and still mismatches.
3. **Item under-play (columns 0-3 via play consequences) is the other survivor**,
   −22% to −26%, consistent with E28c/E28d's "consequence valuation" residual.
4. Gate 1's pre-registered bars and the `conseq − conseq_shuf` read are
   **UNCHANGED** — this addendum re-weights the motivation, not the test.

**Two method findings, both keepers:**

- **The vbeam oracle really was a lower bound, but only mildly.** Root
  disagreement 0.380 (vbeam) → **0.406** (rbeam), +0.026 paired. So E33's
  self-declared caveat is confirmed in sign, and quantified as small: one
  genuine reply ply shifts the divergence instrument by under 3pp. The
  reply-aware oracle also wants MORE trades (0.860 vs 0.814/turn) and LESS face
  (share 0.524 vs 0.543) than the 1-ply oracle — the depth-prefers-board-control
  direction E33 identified, now visible as a dose.
- **Missed-lethal is byte-identical across the two arms (16/356 = 0.045).** It
  comes from `lguard.find_lethal`, which is oracle-independent, so this is a
  built-in consistency check on the new switch and it passes.

Caveat on the 0.045: it is measured on 356 lethal-turns from this 360-game run,
against E37c's ~1600 lethal-turns at 0.0873-0.1047 on the e37 chain nets. Do not
read 0.045 as a gen7 improvement over those — different chain endpoint, different
n, different opponent exposure. It needs the 1500-game protocol to compare.

## VERDICT (2026-07-24) — Gate 1 CLOSES; Gate 2 NOT opened

Run: `scripts/e38_conseq_bc.py`, 3 seeds, practicum `mcts:100` / 156 games /
seed 0 / n=35,320 (E30's exact recipe; E30 had n~35,150), ~7,000 held-out records
per seed. `runs/e38/conseq_bc5.json`.

| arm | multi-action | overall | whole-turn | params |
|---|---:|---:|---:|---:|
| `factored` (E30 baseline) | 0.3761 | 0.4020 | 0.2407 | 130,459 |
| `conseq` (per-action MLP) | 0.4024 | 0.4254 | 0.2432 | 160,257 |
| `conseq_shuf` | 0.3047 | 0.3285 | 0.1882 | 160,257 |
| `factored_plus` (dense head + per-action conseq term) | 0.3966 | 0.4220 | 0.2578 | 130,972 |
| `factored_plus_shuf` | 0.3757 | 0.4005 | 0.2432 | 130,972 |

**E30 anchor reproduced: `factored` 0.3761 vs E30's published 0.373** (overall
0.4020 vs 0.402, whole-turn 0.2407 vs 0.247). The harness is faithful.

**The pre-registered read came out AMBIGUOUS and the tiebreakers close it.**
`conseq − conseq_shuf` = **+0.0976** [.0928, .1034, .0967], just under the +0.10
open bar. But that arm carries a **−0.0713 self-handicap** (`conseq_shuf −
factored` = −.069/−.0831/−.0619): the per-action MLP is simply a worse head than
the dense one, so the shuffle delta measures "columns buying back a self-inflicted
loss" and OVERSTATES the practical gain.

`factored_plus` was added to remove that confound — E30's dense head byte-for-byte
plus a small per-action consequence term, i.e. the shape Gate 2 would actually
use. Its own shuffle control lands **exactly on the baseline** (0.3757 vs 0.3761),
confirming the architecture is neutral, so its delta is purely the columns:

| handicap-free read | per seed | mean |
|---|---|---:|
| `factored_plus − factored_plus_shuf` | +0.0180, +0.0180, +0.0267 | **+0.0209** |
| `factored_plus − factored` | +0.0187, +0.0163, +0.0265 | **+0.0205** |

**+0.021 is inside the pre-registered close band (≤ +0.03), on three seeds with
no overlap of the open bar. E38 CLOSES.** Per the kill criteria, the SB3/pointer-
head integration is NOT built.

**Column ablation (seed 0, informational) — the signal is real, board-control
shaped, and NOT a lethal shortcut:**

| columns kept | `conseq` | `conseq − shuf` |
|---|---:|---:|
| all 10 | 0.4009 | +0.0928 |
| **drop `wins_now`+`enables_lethal`** | 0.3986 | **+0.0821** |
| `dphi` only | 0.3481 | +0.0400 |
| board-control (`dphi`,`kills`,`losses`,`overkill`) | 0.3687 | +0.0667 |
| lethal only (`wins_now`,`enables_lethal`) | 0.2916 | +0.0103 |
| `overkill` only | 0.2989 | +0.0162 |

Dropping BOTH lethal columns costs only 0.011 of the delta, and lethal-only is
worth +0.010 alone — so the pre-registered confound (an `mcts:100` teacher always
takes a lethal, so the lethal columns could ace decisions E33 says are not the
win-rate gap) is **ruled out**. `dphi` is the strongest single column; the
board-control four carry ~72% of the delta. No subset reaches the plain baseline.

**What this establishes — a stronger negative than E30's own.** E30 showed the
reactive obs does not separably encode the teacher's choice. E38 shows that
handing the net **the engine's own answer for what every candidate action does** —
board-power delta, damage, kills, losses, overkill, mana, an exhaustive own-turn
lethal solve, and the opponent's lethal-back flag — moves plan-level agreement by
+0.021, a fifth of the bar E30 itself used to justify opening a training arm. The
missing quantity is therefore **not information about immediate consequences**; it
is the multi-turn valuation of those consequences. That is what depth supplies,
and it lines up with Addendum 1's crossover: gen7 is at parity with fair search to
~900 sims and behind at 2250.

**Consequence for the program: the observation direction is now EXHAUSTED for
search-choice recovery** (E28c completed static card features and won; E38
completes simulated per-action consequences and does not). Add it to the
out-of-scope list. Cost/validity bookkeeping, for the record: 209,201 actions
simulated over the practicum, 37,964 lethal probes (18.2% of actions passed the
cheap filter), **0 node-cap hits** at `LETHAL_NODE_CAP=600`; recording overhead
+7% wall clock with byte-identical example counts.

Keepers regardless of the verdict: `locma/envs/conseq.py` (fair, tested,
non-mutating per-action afterstate features — reusable as a diagnostic or a
search-ordering heuristic), the `--conseq` practicum flag, and
`scripts/e38_conseq_bc.py`'s handicap-controlled BC pattern (a per-action arm
needs BOTH a shuffle control AND an un-handicapped baseline arm, or its delta is
uninterpretable — this round would have read as "open Gate 2" without the latter).

## Gate 1 — BC (the load-bearing, decisive-either-way experiment)

Harness: the E30 controlled-BC design (`scripts/e30_plan_bc.py`), same frozen
features, same masked-CE training, same turn reconstruction, same practicum
teacher (`mcts:100`) — so `factored` reproduces E30's 0.373 and anchors the run.
Three arms, and the third is the control that makes this tight:

| arm | head | isolates |
|---|---|---|
| `factored` | `logits = MLP(state)` | E30's baseline; must reproduce ~0.373 |
| `conseq` | `logit(a) = MLP([h(state), conseq(a), fam(a)])` | the columns |
| `conseq_shuf` | identical architecture, columns **permuted across actions within each record** | architecture + marginal column distribution, with the action↔consequence correspondence destroyed |

`conseq_shuf` is the E28-gate-1 lesson applied: a per-action head is a different
function class than a dense 155-way head, so a `conseq` > `factored` gap alone
would confound information with architecture. The pre-registered read is on
**`conseq` − `conseq_shuf`**.

**Pre-registered decision (on multi-action turns, the E30 subset, 3 seeds):**

- `conseq − conseq_shuf >= +0.10` → the columns carry decision-relevant
  information the reactive representation lacks. **Open the PPO arm.** E30's
  representational verdict is scoped down to "the *v0/fx* representation", not
  representation in general.
- `<= +0.03` → the search teacher's choice is not recoverable even from explicit
  1-ply consequences. **E38 closes, and E30's verdict survives a much sharper
  attack** — the obs direction is then properly exhausted, not merely untested.
- between → ambiguous; judge on whole-turn exact match and the per-column
  ablation below.

Secondary (free, informational): per-column drop-one ablation on the winning
arm, to say WHICH consequence carries the signal — and specifically whether
`enables_lethal` alone accounts for it (which would predict a lethal-tail-only
gain and a small win-rate effect, given E33's "tactics is not the gap").

## Gate 2 — PPO arm (runs only if Gate 1 opens)

Pre-registered now so the follow-up cannot drift:

- Obs variant `ac` (opt-in, default paths byte-identical, `token_variant_for_
  space` extended — the fx precedent). Consequence block enters the **pointer
  head**, not the slot tokens: the head is already per-action
  (`logit(a) = MLP([z_src, z_tgt, latent_pi, fam])`), so the block concatenates
  there. Slot tokens cannot express it — an Attack has 7 target codes per board
  slot.
- Warm-start from `depot:e36s22/e36_s22_gen7.zip` and continue the PFSP chain
  2 generations at the s22 regime (1.5M steps/gen, n_envs 12), against a
  **matched-budget control** continued from the same checkpoint with the same
  pool — the E11/E37c budget-confound discipline.
- Primary ruler: **`dmcts:15,150`** (2250 sims), n=400/net. The `rbeam:shared`
  gate is saturated at parity (3-seed pooled 0.509 [.481,.537]) and cannot
  measure this; the hard dmcts rung leaves headroom in both directions. Ruler
  re-pinned in the same round (`scripts/e36_dmcts_ladder.py --rungs`,
  `runs/e38/repin_dmcts.json`).
- Mechanism instruments: `scripts/e33_reactive_vs_search_behavior.py --oracle
  rbeam` (root-disagree / face-share / trade dphi / overkill, now against a
  reply-aware oracle rather than the 1-ply lower bound) and the missed-lethal
  rate.
- Guard-rail: `scripts/e36_exploit.py` — a net handed lethal flags must not
  regress the E10 archetypes (E21 trap discipline).
- Report the pure-reactive number and the `ac` number **separately**, never
  pooled (see the labeling caveat).

## Kill criteria

- Gate 1 in the `<= +0.03` band → E38 closes; record "the obs direction is
  exhausted for search-choice recovery" and do not build the SB3 integration.
- Gate 1 opens but the PPO arm is CI-negative vs the matched control on
  `dmcts:15,150` → the information is decodable but not usable, which would be
  a third instance of E27's "representing != using" and closes the arm with a
  mechanism.
- Any exploit-guard-rail failure kills the arm regardless of ruler movement.

## Non-goals

Learned dynamics models (E15 non-goal, unchanged: a perfect fast simulator
exists). Multi-ply consequence features — one ply only, or the fairness argument
in §Fairness stops holding without determinization. Using an `ac` net as a
search evaluator (E32 territory; the evaluator paths have no GameState and will
raise rather than silently zero-fill).

## Artifacts

`locma/envs/conseq.py` (feature computation), `locma/envs/practicum.py`
(`conseq=` recording), `scripts/e38_conseq_bc.py` (Gate 1 driver),
`runs/e38/conseq_bc*.json`, `runs/e38/practicum-conseq.npz`. Ruler re-pin:
`runs/e38/repin_dmcts.json`, `runs/e38/e33_gen7_{vbeam,rbeam}.json`.
