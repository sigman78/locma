# E4v2: distill the vbeam teacher back into the reactive net (+ expert iteration)

**Date:** 2026-07-03
**Status:** complete — VERDICT: null on every arm (see Results and the worklog)
**Branch:** `feat/vbeam-distill`

## Motivation

E5v1 (`vbeam`) proved the ~0.66 reactive plateau was within-turn plan
composition: beam-searching own-turn sequences over the frozen B0 net scores
**0.863** on the ruler (+0.206 over B0, PR #56). Two questions follow:

1. **E4v2 (distill):** can a *reactive* net absorb part of that +0.206 by
   imitating the planner? The earlier distillation attempts (PR #18, the E4
   table: agreement ~0.37, avg-hard3 ~0.54) all used **cheating teachers**
   (`mcts`/`dmcts` clone the true `GameState`, deciding on the opponent's hand
   and future draws) — an info ceiling no student observation can fix. vbeam
   is the first **fair teacher**: every decision is a deterministic function
   of the public `BattleView` the student sees. The info-ceiling excuse is
   gone; whatever gap remains is representation/optimization, not information.
2. **EXIT (expert iteration):** does the distilled student improve the planner
   itself? The feedback channel is *narrow by construction*: `plan_turn`
   expands **all** legal actions and ranks by critic value alone — the policy
   head enters only through the `would_pass` stop rule (a stop is scoreable
   with V(s) only where the policy's masked argmax is Pass). A student whose
   argmax matches the planner's own choices (including its Pass decisions)
   changes vbeam's stop-eligibility pattern, nothing else. The FVI study
   (E5v2, null) froze this channel deliberately; this experiment moves it.

## Teacher signal

`record_practicum(teacher="vbeam:...", obs_mode="token")` already captures
exactly what we need: one `(token obs, chosen semantic action, legal mask)`
example per non-forced planner decision, including the planner's mid-turn
Pass decisions (recorded whenever Pass is chosen with other actions legal —
the stopping behavior IS part of the imitation target).

**Data per seed s ∈ {0,1,2}** (`runs/vdst-data-s{s}.npz`):
teacher `vbeam:depot:b0/b0_s{s}.zip`, opponents = the 4-opponent zoo
(greedy/scripted/max-guard/max-attack) **+ vbeam self-play**
(`vbeam:depot:b0/b0_s{s}.zip` as opponent), 400 games/opp x 2 seats,
training seeds 20000+500s (disjoint from FVI's 10k+ and the 1M+ eval range).
Expected ~90k examples/seed. Collection via `collect_value_data` (19 workers).

## Arms (all trained per seed on the same data)

| arm | init | trainable | critic | eval |
|-----|------|-----------|--------|------|
| **PH** | b0_s{s} | `mlp_extractor.policy_net` + `action_net` only (frozen precomputed features) | **byte-identical** | reactive AND `vbeam:` drop-in (the EXIT arm) |
| **BC** | scratch | everything (masked CE) | untrained garbage | reactive only (E4 replication with a fair teacher) |
| **FF** | b0_s{s} | everything (masked CE warm start) | drifts (shared extractor) | reactive only |

PH mirrors `train_value_head` exactly (same freeze discipline, opposite
branch): `train_policy_head` in `locma/envs/vbeam_distill.py`, masked CE on
frozen precomputed features, game-level split, val top-1 agreement
before/after (the "before" number = how often B0's argmax already matches the
planner — a free diagnostic of how much signal distillation can add).

## Eval protocol (standard ruler)

Pilot 10x10 gate, then full 40x25 `ceiling-eval` (paired bootstrap, +0.03
threshold), workers=19:

- **Reactive:** candidates `runs/vdst-{arm}_s{0,1,2}.zip` vs baselines
  `depot:b0/b0_s{0,1,2}.zip`. Full run unconditionally (cheap).
- **Planner (EXIT):** candidates `vbeam:runs/vdst-ph_s{0,1,2}.zip` vs
  baselines `vbeam:depot:b0/b0_s{0,1,2}.zip`. Full run if pilot delta > -0.02.

**Round 2 (conditional):** if the EXIT pilot shows any lift (>= +0.01) or the
reactive PH clears +0.03 on the full ruler, re-collect from
`vbeam:runs/vdst-ph_s{s}` (self-play included), fine-tune again (ph2), and
re-run both evals — the compounding test that AZ-lite failed.

## Predictions (written before results)

- Agreement-before (B0 vs planner) well above the 0.37 cheater-teacher
  plateau; agreement-after higher still.
- Reactive students: real lift over 0.657 but short of 0.863 — pattern
  matching should recover chunks of plan composition (common attack orderings,
  stop discipline) but not exact lethal enumeration.
- EXIT arm: small or null — the would_pass channel is narrow, and FVI showed
  the critic side is already a fixed point.

## Results

Overnight run 2026-07-03, 03:56–06:09 (2h13m, 19 workers). ~99k examples/seed,
0 failed games. Full verdicts (40x25 ruler, paired vs B0 / vbeam:B0):

| arm | val agreement | avg-hard3 | delta | 95% CI | verdict |
|-----|---------------|-----------|-------|--------|---------|
| (B0 argmax vs teacher) | 0.444 | 0.660 | — | — | reference |
| PH reactive | 0.492 | 0.645 | -0.015 | [-0.024, -0.007] | ceiling-confirmed |
| BC reactive | 0.502 | 0.562 | -0.098 | [-0.106, -0.091] | ceiling-confirmed |
| FF reactive | 0.529 | 0.662 | +0.002 | [-0.006, +0.010] | ceiling-confirmed |
| vbeam:PH (EXIT) | — | 0.8641 | -0.0001 | [-0.0008, +0.0006] | ceiling-confirmed |

Round 2 skipped by the pre-registered trigger (no EXIT or reactive signal).

**Prediction check:** agreement-before beat the 0.37 cheater plateau as
predicted (0.444), but the reactive-lift prediction was **falsified** — the
fair teacher raised agreement, not play. The EXIT null was predicted, and
landed with remarkable precision; a key part of the mechanism is that
Pass-with-alternatives examples are only ~0.3% of the data and occur exactly
where B0 already passes (the stop rule construction), so the `would_pass`
channel carries zero trainable signal by design.

Full mechanism reading in `docs/worklog.md` (E4v2 entry). Artifacts:
`runs/vdst-data-s{0,1,2}.npz`, `runs/vdst-{ph,bc,ff}_s{0,1,2}.zip`,
`runs/vdst-summary.json`, `runs/vdst-overnight.log`. Orchestration:
`scripts/vdst_driver.py` (idempotent stages; re-running skips completed
steps via the summary file, so the whole night is resumable/reproducible).
