<!-- web/src/components/Play/BattleScreen.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import type { ActionDict, CardState, EventDict, PlayerState } from '../../lib/replay'
  import {
    attackTargets,
    canSummon,
    cardDamage,
    itemTargets,
    lungeDirFor,
    splashesFor,
    type BattlePending,
    type PlayStep,
  } from '../../lib/play'
  import { computeFx } from '../../lib/fx'
  import { popIn } from '../../lib/motion'
  import { nearestTarget, type AimTarget } from '../../lib/aim'
  import { dock } from '../../lib/dock'
  import CardView from '../ReplayViewer/CardView.svelte'
  import Player from '../ReplayViewer/Player.svelte'
  import PointerLine from './PointerLine.svelte'

  export let pending: BattlePending
  export let you: number
  export let events: EventDict[] = []
  export let currentAction: ActionDict | null = null
  export let fxToken = 0
  export let liveStep: PlayStep | null = null
  export let playing = false

  const dispatch = createEventDispatcher<{ act: ActionDict }>()

  $: view = liveStep ? liveStep.view : pending.view
  $: legal = pending.legal
  $: meSeat = you
  $: opSeat = 1 - you
  $: interactive = !playing
  $: actSeat = liveStep ? liveStep.seat : you
  $: fx = computeFx(events, currentAction, actSeat)
  $: splashes = splashesFor(events)

  function send(a: ActionDict) {
    drag = null
    snapId = null
    dispatch('act', a)
  }

  // --- target anchor registry: board slots + opponent face register their DOM node ---
  const anchors = new Map<number | 'face', HTMLElement>()
  function anchor(node: HTMLElement, id: number | 'face') {
    anchors.set(id, node)
    return {
      destroy() {
        if (anchors.get(id) === node) anchors.delete(id)
      },
    }
  }
  function centerOf(el: HTMLElement): { x: number; y: number } {
    const r = el.getBoundingClientRect()
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 }
  }

  // --- drag-to-aim state ---
  type Drag = { kind: 'attack' | 'use'; src: number; from: { x: number; y: number } }
  let drag: Drag | null = null
  let cursor = { x: 0, y: 0 }
  let snapId: number | 'face' | null = null

  function legalIdsFor(d: Drag): number[] {
    return d.kind === 'attack' ? attackTargets(legal, d.src) : itemTargets(legal, d.src)
  }
  // ids legal for the current drag, mapped to anchor keys (-1 → 'face')
  $: legalKeys = drag
    ? new Set<number | 'face'>(legalIdsFor(drag).map((id) => (id === -1 ? 'face' : id)))
    : new Set<number | 'face'>()

  function aimTargets(d: Drag): AimTarget[] {
    const out: AimTarget[] = []
    for (const id of legalIdsFor(d)) {
      const key: number | 'face' = id === -1 ? 'face' : id
      const el = anchors.get(key)
      if (el) {
        const c = centerOf(el)
        out.push({ id: key, x: c.x, y: c.y })
      }
    }
    return out
  }

  function startAttack(e: MouseEvent, c: CardState) {
    if (!interactive) return
    if (attackTargets(legal, c.iid).length === 0) return
    e.preventDefault() // suppress native text-selection / focus during the drag
    drag = { kind: 'attack', src: c.iid, from: centerOf(e.currentTarget as HTMLElement) }
    cursor = { x: e.clientX, y: e.clientY }
    snapId = null
  }
  // hand mousedown: targeted items start a drag; summon / no-target handled on click
  function downHand(e: MouseEvent, c: CardState) {
    if (!interactive) return
    const ts = itemTargets(legal, c.iid)
    if (ts.length === 0) return
    if (ts.length === 1 && ts[0] === -1) return // no-target item → click handles it
    e.preventDefault()
    drag = { kind: 'use', src: c.iid, from: centerOf(e.currentTarget as HTMLElement) }
    cursor = { x: e.clientX, y: e.clientY }
    snapId = null
  }
  function clickHand(c: CardState) {
    if (!interactive || drag) return
    if (canSummon(legal, c.iid)) {
      send({ t: 'summon', id: c.iid })
      return
    }
    const ts = itemTargets(legal, c.iid)
    if (ts.length === 1 && ts[0] === -1) send({ t: 'use', item: c.iid, target: -1 })
  }

  function onMove(e: MouseEvent) {
    if (!drag) return
    cursor = { x: e.clientX, y: e.clientY }
    const best = nearestTarget(cursor.x, cursor.y, aimTargets(drag))
    snapId = best ? best.id : null
  }
  function onUp() {
    if (!drag) return
    const d = drag
    const sid = snapId
    drag = null
    snapId = null
    if (sid === null) return // released on empty → cancel, no action
    const target = sid === 'face' ? -1 : sid
    if (d.kind === 'attack') send({ t: 'attack', a: d.src, target })
    else send({ t: 'use', item: d.src, target })
  }
  function onKey(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      drag = null
      snapId = null
    }
  }

  // line endpoint: snap to the highlighted target's center, else follow the cursor
  $: lineTo = drag
    ? snapId !== null && anchors.get(snapId)
      ? centerOf(anchors.get(snapId) as HTMLElement)
      : cursor
    : null

  function isPlayable(c: CardState): boolean {
    return interactive && (canSummon(legal, c.iid) || itemTargets(legal, c.iid).length > 0)
  }

  $: mePlayer = {
    health: view.me.health, mana: view.me.mana, max_mana: view.me.max_mana,
    damage_counter: 0, bonus_draw: view.me.bonus_draw, deck_count: view.me.deck_count,
    hand: view.me.hand, board: view.me.board,
  } as PlayerState
  $: opPlayer = {
    health: view.op.health, mana: view.op.mana, max_mana: view.op.max_mana,
    damage_counter: 0, bonus_draw: view.op.bonus_draw, deck_count: view.op.deck_count,
    hand: new Array(view.op.hand_count).fill(null), board: view.op.board,
  } as unknown as PlayerState

  const back: CardState = { iid: -999, card_id: 0, atk: 0, def: 0, abilities: '' }
  $: oppBacks = new Array(view.op.hand_count).fill(back)
</script>

<svelte:window on:keydown={onKey} on:mousemove={onMove} on:mouseup={onUp} />

<PointerLine from={drag ? drag.from : null} to={lineTo} />

<div class="battle" class:playing>
  <Player player={opPlayer} name="AI" seat={opSeat as 0 | 1} active={false} {fx} {fxToken} />
  <button
    class="face op"
    class:legaltarget={legalKeys.has('face')}
    class:snapped={snapId === 'face'}
    use:anchor={'face'}
    title="opponent (drag a unit here to attack)">🎯 face</button>

  <div class="hand backs">
    {#each oppBacks as _b, i (i)}<CardView card={back} faceUp={false} />{/each}
  </div>

  <div class="field top">
    {#each view.op.board as c (c.iid)}
      <button class="slot" class:legaltarget={legalKeys.has(c.iid)} class:snapped={snapId === c.iid}
        use:anchor={c.iid} in:popIn>
        <CardView card={c} facing="down" lunge={lungeDirFor(fx, you, opSeat, c.iid)}
          damage={cardDamage(splashes, opSeat, c.iid)} {fxToken} />
      </button>
    {/each}
  </div>

  <hr />

  <div class="field bottom">
    {#each view.me.board as c (c.iid)}
      <button class="slot" class:legaltarget={legalKeys.has(c.iid)} class:snapped={snapId === c.iid}
        class:armed={drag?.src === c.iid} use:anchor={c.iid} in:popIn
        on:mousedown={(e) => startAttack(e, c)}>
        <CardView card={c} facing="up" dim={c.can_attack === false}
          lunge={lungeDirFor(fx, you, meSeat, c.iid)}
          damage={cardDamage(splashes, meSeat, c.iid)} {fxToken} />
      </button>
    {/each}
  </div>

  <div class="hand mine" use:dock={{ enabled: interactive && !drag }}>
    {#each view.me.hand as c (c.iid)}
      <button class="slot" class:playable={isPlayable(c)} class:armed={drag?.src === c.iid}
        in:popIn on:mousedown={(e) => downHand(e, c)} on:click={() => clickHand(c)}>
        <CardView card={c} showAuras={false} />
      </button>
    {/each}
  </div>

  <Player player={mePlayer} name="You" seat={meSeat as 0 | 1} active={true} {fx} {fxToken} />

  <div class="controls">
    <span class="turnno">Turn {view.turn}</span>
    <span class="hint">
      {#if playing}AI is taking its turn…{:else if drag}Drag to a highlighted target — release to confirm, Esc to cancel.{:else}Your turn — drag a unit to attack, drag an item to its target, click to summon, or end turn.{/if}
    </span>
    <button class="endturn" on:click={() => send({ t: 'pass' })} disabled={!interactive}>End Turn ⏭</button>
  </div>
</div>

<style>
  .battle { --card-w: 100px; --card-h: 140px; --gap: 8px; --hand-cols: 8;
    display: flex; flex-direction: column; gap: 8px; align-items: center;
    width: max-content; max-width: 100%; margin: 0 auto;
    background: #15151b; border-radius: 8px; padding: 14px; color: #ddd; }
  .battle.playing { cursor: progress; }
  .field { display: flex; gap: var(--gap); align-items: center; justify-content: center;
    min-height: calc(var(--card-h) + 12px); padding: 6px;
    background: rgba(255, 255, 255, 0.02); border-radius: 6px;
    width: calc(6 * var(--card-w) + 5 * var(--gap) + 16px); }
  .hand { display: flex; gap: var(--gap); justify-content: center; align-items: center; padding: 6px;
    background: #20212b; border: 1px solid #313445; border-radius: 8px;
    width: calc(var(--hand-cols) * var(--card-w) + (var(--hand-cols) - 1) * var(--gap) + 16px);
    min-height: calc(var(--card-h) + 12px); }
  .hand.backs { opacity: 0.85; }
  /* slots are <button>s (keyboard-focusable, matching the Slice A a11y fix) reset to bare wrappers */
  .slot { background: none; border: 2px solid transparent; border-radius: 8px; padding: 2px;
    cursor: pointer; transition: transform 0.12s ease, box-shadow 0.12s ease; }
  .slot:hover { border-color: #4a4f6a; }
  .slot.playable { border-color: #4fd97a; box-shadow: 0 0 9px rgba(79, 217, 122, 0.45); }
  /* the unit/item currently being dragged */
  .slot.armed { border-color: #ffd23d; box-shadow: 0 0 9px rgba(255, 210, 61, 0.5); }
  /* a legal target during an active drag, brighter when the line is snapped to it */
  .slot.legaltarget { border-color: #5aa9ff; box-shadow: 0 0 8px rgba(90, 169, 255, 0.5); }
  .slot.snapped { border-color: #ff5d5d; box-shadow: 0 0 12px rgba(255, 93, 93, 0.85); }
  hr { width: 70%; border: none; border-top: 1px dashed #3a4a3c; margin: 2px 0; }
  .controls { display: flex; gap: 16px; align-items: center; margin-top: 4px; }
  .turnno { color: #ffd23d; font-weight: 700; font-size: 14px;
    background: rgba(255, 210, 61, 0.12); border: 1px solid #ffd23d55;
    border-radius: 10px; padding: 2px 10px; }
  .hint { color: #aaa; font-size: 14px; }
  .endturn { background: #2a2a44; color: #fff; border: 1px solid #4a4f6a;
    border-radius: 4px; padding: 8px 18px; cursor: pointer; font-weight: 600; }
  .endturn:disabled { opacity: 0.5; cursor: default; }
  .face.op { background: #2b1a1a; color: #ffb4b4; border: 1px solid #5a3a3a;
    border-radius: 4px; padding: 4px 12px; cursor: pointer; transition: box-shadow 0.12s ease; }
  .face.op.legaltarget { border-color: #5aa9ff; box-shadow: 0 0 8px rgba(90, 169, 255, 0.5); }
  .face.op.snapped { border-color: #ff5d5d; box-shadow: 0 0 12px rgba(255, 93, 93, 0.85); }
</style>
