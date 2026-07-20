# E31 — draft item-valuation fix (planned)

Status: **planned** (2026-07-19, not yet started). Scope is correctness and
probe coverage, explicitly NOT a strength lever — E17's dose-response closed
draft-side item enrichment for win rate (every dose null-to-negative, even
under the planner).

## Problem (established 2026-07-19, worklog "E29 pre-read" data caveat)

Two independent holes, both draft-side (engine handling of blue items was
audited against the spec and is correct — `_apply_item`, `battle_legal`,
semantic action space):

1. **Sign-blind heuristic.** `GreedyDraftPolicy._score` = `atk + def +
   0.5*kw - 1` — every blue item scores -1..-5 (their effect lives in
   NEGATIVE defense or in hidden fields), so the mcts/dmcts practicum
   teachers drafted ZERO blue items across all recorded practicums (63k+
   decision states, blue in-hand count 0; red near-zero at ~370). Every
   item-behavior probe on those datasets is blind to blue.
2. **Hidden effect fields.** `CardView` does not expose
   `player_hp`/`enemy_hp`/`card_draw`, so even the magnitude-correct
   `_card_value` (balanced) values 4 of the 8 blues (153/154/156/160) at
   ~0, and `encode_draft` (same 15-feature `_card_block`) makes
   153/154/160 exact input aliases for the learned draft — ldraft cannot
   distinguish Healing Potion from Poison even in principle.

Immediate coverage needs no code: record practicums with the 5th-param
draft override (e.g. `dmcts:15,30,,,-4` = balanced with item bonus,
~5.5 items/deck per E17's table).

## Steps

### E31a — per-card value table (data-only) — DONE (2026-07-20)

A `DistilledDraftPolicy` JSON (registry already loads `values`-keyed JSON
via the 5th spec param) with hand-computed true values including hidden
effects: `|atk| + |def| + player_hp + |enemy_hp| + 2*card_draw + kw`
(clamped like `_STAT_CAP`). No code change. Gives a correct reference
heuristic for diagnostics and teacher-side deck generation. Acceptance:
blues draftable at sane rates; spot-check top-value items.

Shipped: `scripts/e31a_value_table.py` -> `runs/e31a_values.json`
(reference, 0.3 blues/deck) + `runs/e31a_diet.json` (+4 item bonus, the
training-deck source: 6.28 items / 1.13 blues / 72% of decks carry blue).
Decimate tops the item list; all 8 blues price 4-6 (were ~0/negative).
Also landed the training-side plumbing this needed:
`_draft_override_policy` loads values-JSON tables and `train-zoo` exposes
`--draft-override`. Consumed by the E28d arm (worklog 2026-07-20).

### E31b — CardView hidden fields + spell-aware heuristic variant

Expose `player_hp`/`enemy_hp`/`card_draw` on `CardView` (additive), add a
NEW named policy (e.g. `greedy2`) with the magnitude+hidden-effect score.
`greedy` and `balanced` defaults stay byte-identical — they are pinned
reproducibility baselines (registry docstring; E17 confirmed balanced's
discount-12 stands). Acceptance: `locma draft-bench` calibration run under
a strong pilot; no behavior change in any existing spec.

### E31c — draft-obs extension + ldraft retrain (GATED on E30)

Extend `encode_draft` with the three hidden fields per offered card
(`DRAFT_OBS_SIZE` 67 -> 76), retrain ldraft at the E18b recipe, standard
verdict protocol vs the current `depot:ldraft`. Do NOT run before E30:
ldraft's win signal comes from battle nets that cannot convert items
(E27/E28 gate-1/gate-2 triple agreement: consequence valuation, not
access), so with today's battle nets it re-learns item avoidance with
better eyesight — expected delta ~0 (E17, E18c). Execute when the E30
turn-plan head (or any arm) gives the battle net item-conversion ability,
or bundle as its prerequisite.

**Weakened further by E28d (2026-07-20).** The E28d arm trained a battle
net on blue-rich decks (the E31a diet source) and found item/blue
conversion did NOT rise vs the item-light-trained e28c — a battle net
saturated in blue training converts blues no better. So even after E31c
gives ldraft the eyesight to draft blues, today's battle nets would not
convert the extra blues into wins. E31c is doubly gated on E30: it needs
both a reason for ldraft to draft blues AND a battle net that can cash
them. Do not run standalone.

## Non-goals

- Mutating `GreedyDraftPolicy` / `BalancedDraftPolicy` defaults (pinned).
- Any draft-side strength claim (closed by E17; E31 wins are correctness,
  probe coverage, and unblocking E30-era item work).
