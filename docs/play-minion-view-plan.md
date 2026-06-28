# Play-mode MinionView — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task. Rough task list mapped onto the spec
> (`docs/play-minion-view-design.md`); each task is TDD-where-logic-allows + a commit.
> M1–M3 are code (subagent-friendly); M4 is build/verify (orchestrator). Branch:
> `feat/play-minion-view`.

**Goal:** Render Play-mode board minions as frameless floating sprites with prominent
Taunt/Ward/Lethal visuals and a flying breakthrough cue. Web frontend only.

**Architecture:** A new `MinionView.svelte` replaces `CardView` for the two Play board fields;
the three aura keywords get dedicated CSS/SVG channels that compose; the breakthrough cue is a
frontend-only flying number driven by the existing event stream (no engine change).

**Tech Stack:** Svelte 5, TypeScript, Vite, Vitest. All web commands run **from `web/`**.

## Global Constraints (every task inherits these)

- **Web frontend only** (`web/`); **no engine/Python changes; zero replay-data impact.**
- **Play board only:** the two fields in `BattleScreen.svelte` use `MinionView`. Play **hand**
  keeps `CardView`; the **ReplayViewer** (`Board.svelte`/`CardView`) is **unchanged**.
- **Sprites are alpha cutouts** — Lethal glow uses `filter: drop-shadow()` on the `<img>`.
- **Auras compose**, fixed z-order back→front: `taunt shield → lethal glow → sprite → ward
  bubble → stat plates + B/C/D pills + 💤 → transient combat overlays`.
- **Breakthrough always animates** — do NOT add a `prefers-reduced-motion` skip (explicit).
- Placeholder SVG/CSS for shield/bubble/glow/blob — keep them **swappable** (a single asset
  per channel, easy to replace with a PNG).
- Verify with **`npm run check`** (svelte-check + tsc) and **`npm test`** (vitest) green, and
  **`npm run build`** succeeds. Reuse existing helpers (`abilities.ts`, `fx.ts`, `play.ts`,
  `motion.ts`) — don't duplicate logic.

---

### M1 — `MinionView.svelte` (frameless board minion) + wire into BattleScreen · spec "MinionView"
**Files:** create `web/src/components/Play/MinionView.svelte`; modify
`web/src/components/Play/BattleScreen.svelte`; extend `web/src/lib/abilities.ts` +
`web/src/lib/abilities.test.ts`

- Add a pure helper to `abilities.ts`: `auraSplit(mask) -> { taunt: boolean, ward: boolean,
  lethal: boolean, pills: AbilityInfo[] }` where `taunt = hasAura(mask,'G')`, `ward =
  hasAura(mask,'W')`, `lethal = hasAura(mask,'L')`, and `pills = abilityList(mask).filter(a =>
  !AURA_KEYWORDS.includes(a.letter))` (i.e. B/C/D only). Unit-test it.
- `MinionView.svelte`: same **props** as the board usage of `CardView`
  (`card, facing, slideX, slideY, flash, hit, dying, dmgDelay, damage, dim, fxToken`) and reuse
  the same logic (`cardName`/`cardMeta`/`artUrl`, the `atkDelta`/`defDelta` buff math,
  `restartAnim`, the slide/flash anim class, the tooltip block copied from `CardView`). New
  **frameless** layout: render only the sprite `<img>` (no `.card` bg/border); **drop the cost
  gem**; **stats** = `atk` bottom-left + `def` bottom-right, **each on its own dark rounded
  mini-plate** (keep `.buffed`/`.reduced` coloring); keep **💤** sleeping overlay, the **B/C/D
  pills** (from `auraSplit().pills`, with the granted-glow), and the transient
  `flash`/`hit`/`damage`/`death-cross` overlays. (Aura *visuals* land in M2 — here just carry
  the `taunt/ward/lethal` booleans into classes, no shield/bubble/glow yet.)
- In `BattleScreen.svelte`, swap `CardView` → `MinionView` in BOTH `.field.top` (`displayOp`,
  `facing="down"`) and `.field.bottom` (`displayMe`, `facing="up"`) loops, passing the same
  props. Leave the hand (`CardView`) and the opponent hand-backs untouched.
- **Tests:** vitest for `auraSplit` (mask `'--D-LG'`-style → taunt/lethal true, ward false,
  pills = [Drain]; mask `'------'` → all false, pills empty; a granted ward `'-----W'` →
  ward true). `npm run check` passes (component type-checks); existing `npm test` stays green.

### M2 — Aura visuals: taunt shield / lethal glow / ward bubble (compose) · spec "Aura keyword visuals"
**Files:** modify `web/src/components/Play/MinionView.svelte` (+ a small inline/asset SVG for
the shield)

- **Taunt (G):** an SVG shield element behind the sprite (`z` below the `<img>`), sized to the
  **full card slot** (`var(--card-w)`×`var(--card-h)`, centered) so it stays within the slot
  footprint (no neighbor spill). Placeholder SVG (simple heater-shield path, muted blue
  `#5aa9ff`); keep it as one swappable element.
- **Lethal (L):** `filter: drop-shadow(0 0 6px #4fd97a) drop-shadow(0 0 2px #4fd97a)` on the
  sprite `<img>` (static; follows the cutout alpha). Must coexist with any slide/attack
  filters already applied — apply on the img, not the wrapper, to avoid clobbering.
- **Ward (W):** a semi-transparent light-blue bubble div over the sprite but **under** the
  stats/pills (`z` between sprite and stats): thin bright rim + faint radial fill (kept light,
  `mix-blend-mode: screen` like the old `.ward-tint`), with a **slow ~2s pulse** keyframe
  (opacity/scale breathe). Replaces the old `.ward-tint` look.
- Confirm the z-order composes: a minion with G+W+L shows shield behind, green silhouette glow,
  bubble in front of the sprite, stats/pills on top — all at once.
- **Tests:** `npm run check` + existing `npm test` green; the aura booleans (from M1's
  `auraSplit`) drive the three channels. Visual correctness is the user's in-browser playtest.

### M3 — Breakthrough flying-number cue · spec "Breakthrough cue"
**Files:** add a detection helper + test to `web/src/lib/play.ts` +
`web/src/lib/play.test.ts`; modify `web/src/components/Play/BattleScreen.svelte`

- Detection helper (pure, unit-tested): `breakthroughHit(action, splashes, actSeat) -> { amount:
  number } | null` — returns the overflow when `action?.t === 'attack' && action.target !== -1`
  (a minion target) **and** `faceDamage(splashes, 1 - actSeat) != null`; else `null`. (A direct
  face attack has `target === -1` → no fire.)
- In `BattleScreen.onStep()` (the per-step FX director): compute `breakthroughHit(currentAction,
  splashes, actSeat)`. If non-null, capture the **source rect** = the target minion's anchor
  (`rectOfKey(currentAction.target)`) and the **destination rect** = the defender face anchor
  (`rectOfKey(actSeat === you ? 'face' : 'face-me')`), and render a transient **flying red
  number** element (`-{amount}`) that springs source→destination over **~300ms** with `backOut`
  easing (reuse `svelte/easing` `backOut` as in `motion.ts`) and a **short fading trail**,
  fired at the attack **apex** (start it alongside the existing slide; the FX window is already
  opened by `pulse`). Keep the existing face number (it is the landing state); the flight is the
  number's entrance.
- Keep it swappable/minimal: one positioned element + a CSS trail; clean it up after the
  flight (mirror how `slideMap`/`flashSet` are cleared after `HOLD_MS`).
- **Tests:** vitest for `breakthroughHit` — fires with the right `amount` on (attack + minion
  target + defender face splash); returns `null` for a direct face attack (`target === -1`);
  returns `null` when no face splash. `npm run check` + `npm test` green. The visual (trail,
  springiness, apex timing) is the user's in-browser playtest.

### M4 — Verify + handoff · spec "Testing & acceptance" [orchestrator]
- From `web/`: `npm run check` (svelte-check + tsc clean), `npm test` (vitest all green),
  `npm run build` (succeeds). Report any type/test failures.
- Hand off for the user's in-browser playtest (`locma serve` → Play): a board with
  Taunt/Ward/Lethal minions (compose), and a breakthrough attack (B creature over-killing a
  blocker) to see the flying number. Note the placeholder assets are swappable.

---

**Spec coverage check:** "MinionView" (frameless, stat mini-plates, drop cost, keep 💤, B/C/D
pills, transient overlays, tooltip) → M1; "Aura keyword visuals" (taunt shield / lethal glow /
ward pulse, compose, z-order) → M2; "Breakthrough cue" (detection + flying number, ~300ms
springy, short trail, keep face number, always animate) → M3; scope (Play board only, hand +
ReplayViewer unchanged) → M1 + Global Constraints; "Testing & acceptance" → M1/M3 unit tests +
M4 build/verify + user playtest. No engine change (Global Constraints). No spec item unmapped.
