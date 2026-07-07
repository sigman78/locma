<!-- web/src/components/ReplayViewer/Board.svelte -->
<script lang="ts">
  import { onDestroy } from 'svelte'
  import { flip } from 'svelte/animate'
  import type { CardState, Snapshot } from '../../lib/replay'
  import type { Fx } from '../../lib/fx'
  import { animate as fxWindow, deathFx, spring } from '../../lib/motion'
  import { mergeDisplayBoard } from '../../lib/stepfx'
  import MinionView from '../Play/MinionView.svelte'
  import Hand from './Hand.svelte'
  import Player from './Player.svelte'

  export let snapshot: Snapshot
  export let nameA: string // seat 0
  export let nameB: string // seat 1
  export let fx: Fx | null = null
  export let fxToken = 0
  // minions that died this step, retained so the cross shows before the unit leaves.
  export let dying: { seat: number; card: CardState; index: number }[] = []
  $: p0 = snapshot.players[0]
  $: p1 = snapshot.players[1]

  // freshly-drawn hand cards to glow this step (only the owning seat's hand)
  const NO_DRAWN = new Set<number>()
  $: drawn0 = fx?.drawn && fx.drawn.seat === 0 ? new Set(fx.drawn.iids) : NO_DRAWN
  $: drawn1 = fx?.drawn && fx.drawn.seat === 1 ? new Set(fx.drawn.iids) : NO_DRAWN

  // how long a dead unit lingers (showing the red cross) before its removal plays.
  const CROSS_MS = 300

  // --- death retention ---
  let retained: { seat: number; card: CardState; index: number }[] = []
  let dyingSet = new Set<number>()
  let timers: ReturnType<typeof setTimeout>[] = []

  // a fresh `dying` array (new ref each forward step; [] on seek/back) drives the cross:
  // retain the units, then drop each after CROSS_MS so its out:deathFx removal plays.
  function retain(d: typeof dying) {
    timers.forEach(clearTimeout)
    timers = []
    retained = d.map((x) => ({ ...x }))
    dyingSet = new Set(d.map((x) => x.card.iid))
    for (const x of d) {
      const id = x.card.iid
      timers.push(
        setTimeout(() => {
          retained = retained.filter((r) => r.card.iid !== id)
          dyingSet.delete(id)
          dyingSet = dyingSet
        }, CROSS_MS),
      )
    }
  }
  $: retain(dying)
  onDestroy(() => timers.forEach(clearTimeout))

  // board to render = the settled snapshot board with retained dying units re-inserted
  // at their original slots (so a mid-row death doesn't shuffle survivors).
  // `retained` is referenced directly here so Svelte re-runs these when it changes.
  $: display0 = mergeDisplayBoard(
    p0.board,
    retained.filter((r) => r.seat === 0).map((r) => ({ card: r.card, index: r.index })),
  )
  $: display1 = mergeDisplayBoard(
    p1.board,
    retained.filter((r) => r.seat === 1).map((r) => ({ card: r.card, index: r.index })),
  )

  function dmg(seat: number, iid: number): number | null {
    const s = fx?.splashes.find((x) => x.seat === seat && x.target === iid && !x.fatal)
    return s ? s.amount : null
  }
  // brief red overlay on any minion that lost HP this step (combat: attacker + defender)
  const hitFlash = (seat: number, iid: number): boolean =>
    !!fx?.splashes.some((x) => x.seat === seat && x.target === iid && x.amount > 0)
  // Fixed attack-nudge for the lunging minion, reusing MinionView's slide animation
  // (locma-slide peaks at 60% of the vector ≈ 14px, matching the old lunge). Seat 0
  // sits at the bottom and nudges up (negative); seat 1 at the top nudges down.
  const SLIDE = 24
  function slideFor(seat: number, iid: number): number {
    if (fx?.lunge && fx.lunge.seat === seat && fx.lunge.iid === iid) {
      return seat === 0 ? -SLIDE : SLIDE
    }
    return 0
  }
</script>

<div class="board">
  <Player player={p1} name={nameB} seat={1} active={snapshot.current === 1} {fx} {fxToken} />
  <Hand cards={p1.hand} faceUp={true} active={snapshot.current === 1} tipDir="below"
    drawnIids={drawn1} {fxToken} />
  <div class="battlefield">
    <div class="field top" class:active={snapshot.current === 1}>
      <!-- in:spring = summon spawn-drop; flip = survivors glide to make room /
           close a gap. Both gated by the forward-step window so seeks stay instant. -->
      {#each display1 as c (c.iid)}
        <div in:spring out:deathFx animate:flip={{ duration: $fxWindow ? 220 : 0 }}>
          <MinionView card={c} slideY={slideFor(1, c.iid)} damage={dmg(1, c.iid)}
            hit={hitFlash(1, c.iid)} dying={dyingSet.has(c.iid)} dmgDelay
            dim={c.can_attack === false} facing="down" {fxToken} />
        </div>
      {/each}
    </div>
    <hr />
    <div class="field bottom" class:active={snapshot.current === 0}>
      {#each display0 as c (c.iid)}
        <div in:spring out:deathFx animate:flip={{ duration: $fxWindow ? 220 : 0 }}>
          <MinionView card={c} slideY={slideFor(0, c.iid)} damage={dmg(0, c.iid)}
            hit={hitFlash(0, c.iid)} dying={dyingSet.has(c.iid)} dmgDelay
            dim={c.can_attack === false} facing="up" {fxToken} />
        </div>
      {/each}
    </div>
  </div>
  <Hand cards={p0.hand} faceUp={true} active={snapshot.current === 0} tipDir="above"
    drawnIids={drawn0} {fxToken} />
  <Player player={p0} name={nameA} seat={0} active={snapshot.current === 0} {fx} {fxToken} />
</div>

<style>
  .board { display: inline-flex; flex-direction: column; gap: 10px; padding: 14px;
    background: #15151b; border-radius: 8px;
    --card-w: 108px; --card-h: 150px; --gap: 8px; --board-cols: 6; --hand-cols: 8; }
  /* the arena: visually distinct from the players' hands */
  .battlefield { display: flex; flex-direction: column; align-items: center; gap: 6px;
    padding: 10px 12px; border-radius: 10px;
    background:
      radial-gradient(ellipse at 50% 50%, rgba(120, 60, 30, 0.14), transparent 70%),
      #0b0e0c;
    border: 1px solid #2c3a2e;
    box-shadow: inset 0 0 24px rgba(0, 0, 0, 0.6); }
  .field { display: flex; gap: var(--gap); align-items: center; justify-content: center;
    padding: 6px; background: rgba(255, 255, 255, 0.015); border-radius: 6px;
    width: calc(var(--board-cols) * var(--card-w) + (var(--board-cols) - 1) * var(--gap) + 12px);
    min-height: calc(var(--card-h) + 12px); transition: background 0.2s; }
  /* faintly warm the active player's half of the arena */
  .field.active { background: rgba(255, 210, 61, 0.07); }
  .field > div { flex: 0 0 auto; }
  hr { width: 70%; align-self: center; border: none;
    border-top: 1px dashed #3a4a3c; margin: 0; }
</style>
