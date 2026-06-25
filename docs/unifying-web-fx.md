# Unifying / porting play-mode FX to the replay view

Context: the human play view (`web/src/components/Play/BattleScreen.svelte`) gained
cosmetic combat FX — card draw, attack slide, damage flash, death cross — that the
replay view (`web/src/components/ReplayViewer/`) lacks. This note records a design
grilling about how to share them, and the decision to ship a small copy-port first.

## What is already shared (good news)

Both views already use the same `CardView.svelte`, `Player.svelte`, `Hand.svelte`,
`lib/motion.ts` (`dealIn`/`deathFx`/`spring`/`pulse`/`animate`), `lib/fx.ts`
(`computeFx`), and the keyframes in `app.css`. Critically:

- `CardView` **already accepts every FX prop**: `slideX/slideY`, `lunge`, `flash`,
  `hit`, `dying`, `damage`, `dmgDelay`. `sliding` (measured slide) wins over `lunge`.
- Splash/damage computation is already unified — `play.ts:splashesFor` is literally
  `computeFx(events, null, 0).splashes`.
- `lib/stepfx.ts` (`planStepFx`, `mergeDisplayBoard`) is already data-shape agnostic
  (takes `action, events, rectOf, fallbackDy`), so it is reusable as-is.
- `Hand.svelte` / `Player.svelte` are **replay-only** (Play renders its hand and faces
  inline), so they can be changed without touching Play.

The real divergence lives in exactly one component per side: `BattleScreen.svelte`
(rich FX director) vs `Board.svelte` (lightweight FX lookup).

## Decisions from the grilling (the full-unify path)

These were agreed before scoping down. They remain the target if/when a full unify is
revisited.

1. **Unify, not port** — share one FX engine rather than maintain two.
2. **Share the director *logic*, keep markup per-view** — the markup genuinely differs
   (Play has drag buttons + faceplates; Replay has plain slots + `Player` faces +
   `ActionLog`). The thing that drifts is the FX state machine, so share that; do not
   force the read-only replay board to carry Play's drag interactivity.
3. **Timing: fixed cadence ≥ a shared `STEP_MS`** — the director owns the envelope
   constants (`SLIDE`, `CROSS_MS`, `REMOVAL`, derived `STEP_MS ≈ 850`). Replay autoplay
   derives its interval from `STEP_MS` instead of a hardcoded 600ms, and pulses long
   enough that `deathFx` sees `animate === true` when removal starts. Mirrors Play's
   `HOLD_MS` model (no completion-callback gating). Kills the "load-bearing, keep in
   sync" manual contract in `BattleScreen`.
4. **Card draw: animate both players, on the frame boundary** — replay records full
   hands for both seats. A keyed `{#each (c.iid)}` + `in:dealIn` (already gated by
   `animate`) fires exactly once when a new `iid` enters a hand, only on forward steps,
   never on load/seek. No draw-event plumbing. Needs `perspective` added to `Hand` for
   the `rotateY` flip to read in 3D. Accept that the draw appears on the
   decision-point frame where the card first shows, not a synthesized "draw moment".
5. **Full parity (measured slides), not a fallback nudge** — the snapshot becomes the
   director's *input*; replay's `Board` renders the director's **merged display board**
   (so a dying minion is retained at its original slot during the cross, and slides are
   measured against the still-current DOM). Replay registers `use:anchor` on every slot
   and wraps each `Player` face in a seat-keyed face anchor.
6. **Store-factory director, seat-absolute, `bottomSeat` param** — `createDirector({ bottomSeat })`
   returns stores (`display`, `slides`, `flashes`, `dying`), an `anchor` action, and an
   `onStep(action, events, actingSeat)` method, one instance per view. Replay is
   seat-absolute; **Play is rewired** to feed `boards[you]=view.me.board`,
   `boards[1-you]=view.op.board` and render `display[you]` at the bottom. Accepted that
   this changes working Play code (contained).
7. **Refactor safely** — characterization tests for the extraction (not red-first TDD),
   a **visual parity gate for Play** (CI only runs ruff + pytest; the frontend visual
   check is the real gate), then test-first for the new replay behaviour. Stages:
   (1) extract director + rewire Play, prove identical; (2) wire replay to the director;
   (3) card draw.

## Decision: ship a small copy-port first

The full unify above is correct but heavy. For now we port only the **missing basic
minion effects** into the replay `Board` with no director/factory, no Play changes, and
no card draw:

- **Damaged** — thread the existing `hit` prop into replay `CardView`s from
  `fx.splashes` (`amount > 0`), with `dmgDelay` to land the flash with the number.
- **Die** — show the red death cross + a brief retention before removal. `ReplayViewer.advance()`
  computes the dying cards' `CardState` + board index from the **previous** frame's
  snapshot (the dead minion is still present there) and passes them to `Board`, which
  re-inserts them via `mergeDisplayBoard`, renders them with `dying=true` for `CROSS_MS`,
  then drops them so the existing `out:deathFx` fade plays.
- **Attack** — keep the existing `lunge` up/down (replay already has it); measured
  slide-to-target is deferred to the full unify.

Contained to `Board.svelte` + `ReplayViewer.svelte`; reuses `mergeDisplayBoard`,
`deathFx`, `CROSS_MS`, and `CardView`'s `dying`/`hit`/`dmgDelay` props.
