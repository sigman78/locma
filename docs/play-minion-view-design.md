# Play-mode MinionView + keyword visuals + breakthrough cue — design

**Date:** 2026-06-27
**Status:** approved (sharpened via a grilling session), entering plan
**Branch:** `feat/play-minion-view` (off `main`)

## Goal

Make Play-mode board minions read as **creatures, not cards**: drop the card frame to show the
transparent sprite cutout, give the three "aura" keywords (Taunt / Ward / Lethal) distinct
prominent visuals, keep the rest as pills, and add a flying breakthrough cue. Web frontend only
(`web/`, Svelte 5). **No engine changes; zero replay-data impact.**

## Established facts (verified)

- The card art in `locma/data/assets/NNN.png` are **RGBA alpha cutouts** (285×400, ~75–89%
  transparent, corners alpha 0) — the creature floats on transparency. Dropping the card
  background yields a free-floating sprite, and a silhouette glow is possible via
  `filter: drop-shadow()` (it follows the alpha).
- Breakthrough overflow is **already** in the event stream: `battle.py:353-356` emits a
  `{t:'damage', target:'face', amount}` event exactly when `attacker.has("B") && not warded &&
  overflow > 0`. The frontend already renders this as a face number (`play.faceDamage`). So the
  cue is pure frontend, inferable from existing events, and works on old replays.

## Scope

**In:** a new `MinionView.svelte` for Play **board** minions; the three aura visuals; the
breakthrough flying-number cue; logic unit tests.

**Out:** Play **hand** keeps `CardView` (cards in hand still look like cards). The
**ReplayViewer** board is **unchanged** (`CardView`). No engine/asset-pipeline changes (I draft
placeholder SVG/CSS for the shield/bubble/glow/blob; real PNGs are swapped in later by the user).

## MinionView (board minion rendering)

Replaces `CardView` for the two Play fields (`displayOp` top, `displayMe` bottom) in
`BattleScreen.svelte`. Reuses the existing logic helpers (`abilityList`, `hasAura`, the
buff/debuff delta computation, `restartAnim`, slide/flash/hit/dying/damage props) but a new,
frameless layout:

- **No frame/background** — render only the sprite cutout (`artUrl(card.card_id)`), floating.
  (Same aspect ratio as the slot, so no cropping.)
- **Stats:** `atk` bottom-left, `hp/def` bottom-right, **each on its own small dark rounded
  mini-plate** for legibility over arbitrary art. Keep the buff/debuff coloring (green buffed /
  red reduced) and the granted-ability glow.
- **Drop the cost gem** (meaningless once in play).
- **Keep the 💤 sleeping** (summoning-sick) overlay.
- **B / C / D keywords → pill chips** beside the minion (as today, with the granted-glow for
  in-play-granted abilities).
- **Transient combat overlays unchanged:** hit-flash, `−damage` number, ✕ death-cross.
- Keep the hover tooltip (printed-card detail) — it's still useful on board.

## Aura keyword visuals — compose simultaneously

All three can be present at once. **z-order back→front:**

1. **Taunt (G)** — an **SVG shield** behind the sprite, sized to the **full card slot** (max
   card size; stays within the slot footprint → no neighbor spill). Placeholder SVG drafted in
   this work; swappable for a PNG later.
2. **Lethal (L)** — a **static** green silhouette glow: `filter: drop-shadow(0 0 ~6px
   #4fd97a)` on the sprite `<img>` (follows the cutout alpha). Replaces the old rectangular
   `.card.lethal` outline.
3. **The sprite.**
4. **Ward (W)** — a light-blue, semi-transparent "magic barrier" **bubble over the sprite**
   (thin rim + faint fill, kept light so it doesn't mute the sprite), with a **slow ~2s
   pulse**. Replaces the old `.ward-tint`. Renders **under** the stat plates / pills so the
   numbers stay readable.
5. **Stat plates + B/C/D pills + 💤.**
6. **Transient combat overlays** (top).

## Breakthrough cue (flying number)

- **Trigger** (frontend-only): a step where `currentAction.t === 'attack' && action.target !==
  -1` (a minion target) **and** there is a `target:'face'` splash for the defender seat
  (`faceDamage(splashes, 1 - actSeat) != null`). The overflow `amount` is that splash.
- **Animation:** the overflow damage **number itself** is a red blob that **flies from the
  defender minion (the attack target) into the opponent's (defender's) face**, with a **short
  fading trail**, a **fast ~300ms springy** motion, fired at the **attack apex** (mid-slide of
  the attacker). It **lands as the face number** — i.e. the existing face number arrives via
  this flight rather than popping statically in place (no separate blob *plus* static number).
- **Fixed blob size** (the number conveys magnitude; no size-scaling with overflow).
- **Always animates** — do **not** add a `prefers-reduced-motion` skip (explicit decision).
- Anchors already exist in `BattleScreen` (`rectOfKey` for a minion `iid`, and `'face'` /
  `'face-me'` for the two player faces); the cue uses the target minion rect → the defender's
  face rect.

## Testing & acceptance

- **Unit-test the logic** in the existing `web/src/lib/*.test.ts` (vitest) style: the
  breakthrough-detection predicate (attack + minion target + defender face splash → fire, with
  the right amount; a *direct* face attack `target===-1` does **not** fire) and the
  aura→visual mapping (G/L/W → the right channels; B/C/D → pills).
- **Visuals are playtested in-browser by the user** (`locma serve` → Play). No GIF.
- Keep `npm run check` (svelte-check + tsc) and `npm test` green; `npm run build` succeeds.

## Files

- `web/src/components/Play/MinionView.svelte` — new board-minion component.
- `web/src/components/Play/BattleScreen.svelte` — use `MinionView` for the two fields; add the
  breakthrough flying-number cue (anchors → projectile).
- `web/src/lib/abilities.ts` — already has `AURA_KEYWORDS = ['G','L','W']`; reuse.
- `web/src/lib/play.ts` / `fx.ts` / `stepfx.ts` — a small breakthrough-detection helper +
  its unit test (placement decided in the plan).
- Placeholder assets: taunt shield SVG (+ ward bubble / lethal glow / breakthrough blob as
  CSS/SVG), all swappable.
