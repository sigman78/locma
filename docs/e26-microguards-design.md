# E26 — play-time micro-guards for the reactive rung (+ edraft full confirm)

_Pre-registered 2026-07-13, before any run. Branch `feat/e26-microguards`._

## Question

The training-side absorption program is closed (E15) and deeper search is the
one confirmed planning lever (E22-E24) — but the user constraint here is FAIR
and PREFERABLY NOT DEEP-SEARCHING improvements to the deployed policies. Two
zero-training play-time levers were recorded and never executed, plus one
draft-side confirm:

1. **Lethal guard** (E14a finding 3a, repeated in the E15 closure as a
   "surviving practical item"): the reactive net fails to convert **18.3%** of
   engine-verified forced wins. A cheap exhaustive own-turn lethal solver — no
   net, no multi-turn search — should close most of those at negligible cost.
2. **Reactive policy ensemble**: E8 showed mean-of-critics ensembling is the
   single biggest zero-training planner gain (+0.036). The POLICY-head analog
   (mean of masked action distributions of `b0k_s0|s1|s2`, argmax) has never
   been tried on the reactive rung.
3. **edraft full confirm** (E20 open item): the zero-inference draft heuristic
   matched `ldraft` at pilot scale only; run the full ruler against the
   deployed `ldraft` halves on both rungs.

## New machinery

- `locma/policies/lguard.py`:
  - `find_lethal(state, node_cap) -> (line | None, exhausted)` — exhaustive
    DFS over own-turn non-Pass action sequences on `_clone_battle` forward
    states, returning the first action sequence that reaches
    `Phase.ENDED` with `winner == seat`. View-dedup collapses order
    permutations; loss-terminals are pruned; `node_cap` bounds worst-case
    cost (cap hit -> `(None, False)`). Fairness class identical to `vbeam`:
    own-turn only, never applies `Pass`, deterministic forward model.
  - `LethalGuardBattlePolicy(inner)` — wrapper: plays a found lethal line to
    completion; otherwise delegates to `inner`. Per-turn negative cache: if
    the DFS *exhausted* without a win at some decision, every later state in
    the same turn is in the explored closure, so the guard skips re-searching
    until `view.turn` changes. Counters (`stats` dict) for the mechanism
    probe: decisions, searches, activations, cap hits, nodes.
- `MaskablePPOEnsembleBattlePolicy` (`locma/policies/ppo.py`): mean of the
  members' masked policy distributions, argmax, `index_to_action`.
  Deterministic.
- Registry: `ppo:` model param accepts `|`-separated paths (ensemble), same
  idiom as `vbeam:`; new `lppo:model[,draft[,node_cap]]` spec = `ppo:` +
  `LethalGuardBattlePolicy` (model may be `|`-separated). Old specs
  byte-identical.

## Arms (all on the standard reactive ruler: paired avg-hard3 vs HARD3, common anchors)

Baselines everywhere = the reactive recipe of record
`ppo:depot:b0k/b0k_sX.zip,depot:ldraft/ldraft_sX.zip` (0.791).

| arm | candidates (X = 0,1,2) |
|---|---|
| A `lguard` | `lppo:depot:b0k/b0k_sX.zip,depot:ldraft/ldraft_sX.zip` |
| B `ens` | `ppo:b0k_s0\|b0k_s1\|b0k_s2,depot:ldraft/ldraft_sX.zip` |
| C `lens` | `lppo:b0k_s0\|b0k_s1\|b0k_s2,depot:ldraft/ldraft_sX.zip` |

Stage E (draft side, both rungs, full 40x25 directly — pure eval, no gate):

- reactive: `ppo:b0k_sX,depot:edraft/e20-elicit-fit.json` vs RoR pair.
- planner: `vbeam:<shared ens x3>,8,20,depot:edraft/...` vs
  `vbeam:<shared ens x3>,8,20,ldraft_sX` (the 0.978 RoR).

## Protocol and gates (E19 pattern)

- Pilot 10x10 per arm at 34M anchors; **full 40x25** (same 34M range) iff
  pilot `ci_hi > 0`; **fresh confirm** 40x25 at 35M iff full `ci_lo > 0`.
- Promotion candidate iff full AND confirm `ci_lo > 0` (dominance swap;
  the +0.03 headroom bar reported separately, per E11/E12 precedent).
- Mechanism probe (serial, 200 games `lguard` arm vs HARD3+boardkeep,
  seeds 36M): activations/game, guard-changed-the-move rate (vs inner
  argmax), cap-hit rate, node counts, wall s/game overhead.
- Ordering: A/B/C pilots first (cheap), then fulls, then confirms, then
  probe, then stage E.

## Predictions

- A: positive but small (+0.005..+0.02) — missed lethals are 1.4% of turns
  and the position is usually already winning; conversion to win-rate is
  partial. Cost negligible (cap-bounded DFS, no net).
- B: unknown sign — prob-averaging reduces policy-head noise, but E8's
  mechanism (sibling-ordering de-noise) lives in the critic; the policy analog
  may be null. Cheap to settle either way.
- C: approx A+B if both are real and mechanisms are disjoint.
- E: edraft retains most of ldraft's gain at full scale (pilot said parity on
  the planner rung); a CI-clean deficit vs ldraft on the reactive rung is the
  expected residual (~-0.05 vs the net drafter, per E20's 67% recovery).

## Seed ledger

1M-33M used through E25. E26: 34M primary, 35M confirm, 36M probe.
