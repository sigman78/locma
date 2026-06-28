<!-- web/src/components/Play/BattleScreen.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import type { ActionDict, CardState, EventDict, PlayerState } from '../../lib/replay'
  import {
    attackTargets,
    canSummon,
    cardDamage,
    itemTargets,
    splashesFor,
    type BattlePending,
    type PlayStep,
  } from '../../lib/play'
  import { computeFx } from '../../lib/fx'
  import { spring, dealIn, deathFx } from '../../lib/motion'
  import { mergeDisplayBoard, planStepFx, type RectOf } from '../../lib/stepfx'
  import { nearestTarget, type AimTarget } from '../../lib/aim'
  import { dock } from '../../lib/dock'
  import CardView from '../ReplayViewer/CardView.svelte'
  import MinionView from './MinionView.svelte'
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

  // Timing is load-bearing and must stay in sync with Play.svelte: pulse(700) keeps the
  // animate window open past CROSS_MS so out:deathFx plays; HOLD_MS matches playSequence's holdMs.
  const CROSS_MS = 300 // red cross shows this long, then the unit is dropped (removal plays)
  const HOLD_MS = 850 // matches Play's per-step hold
  const FORWARD = 46 // fallback vertical slide when a target rect is unavailable

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

  // --- anchor registry: board slots + both faces register their DOM node.
  // 'face' = the opponent (top) face (also the drag target); 'face-me' = the
  // human's own (bottom) face, needed so an AI→human-face attack can slide down. ---
  type AnchorKey = number | 'face' | 'face-me'
  const anchors = new Map<AnchorKey, HTMLElement>()
  function anchor(node: HTMLElement, id: AnchorKey) {
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
  const rectOfKey = (key: AnchorKey): { cx: number; cy: number } | null => {
    const el = anchors.get(key)
    if (!el) return null
    const c = centerOf(el)
    return { cx: c.x, cy: c.y }
  }
  const rectOf: RectOf = (key) => rectOfKey(key)

  // --- FX director state ---
  let displayMe: CardState[] = []
  let displayOp: CardState[] = []
  const retained = new Map<number, { seat: number; card: CardState; index: number }>()
  let dyingSet = new Set<number>()
  let slideMap = new Map<number, { dx: number; dy: number }>()
  let flashSet = new Set<number | 'face'>()
  let lastToken = -1

  function syncDisplay() {
    const ret = (seat: number) =>
      [...retained.values()]
        .filter((r) => r.seat === seat)
        .map((r) => ({ card: r.card, index: r.index }))
    displayMe = mergeDisplayBoard(view.me.board, ret(meSeat))
    displayOp = mergeDisplayBoard(view.op.board, ret(opSeat))
  }

  function onStep() {
    // measure on the still-current DOM (the new board has not rendered yet).
    // 'face' means the DEFENDER's face: the human attacks the op (top) face,
    // the AI attacks the human's own (bottom) 'face-me'.
    const fwd = actSeat === you ? -FORWARD : FORWARD
    const stepRectOf: RectOf = (key) =>
      key === 'face' ? rectOfKey(actSeat === you ? 'face' : 'face-me') : rectOfKey(key)
    const plan = planStepFx(currentAction, events, stepRectOf, fwd)
    slideMap = new Map(plan.slides.map((s) => [s.iid, { dx: s.dx, dy: s.dy }]))
    flashSet = new Set(plan.flashes)
    // retain dying units (pull their CardState + original board index from what is shown)
    const stepDying: number[] = []
    for (const d of plan.dying) {
      const board = d.seat === meSeat ? displayMe : displayOp
      const index = board.findIndex((c) => c.iid === d.iid)
      if (index >= 0) {
        retained.set(d.iid, { seat: d.seat, card: board[index], index })
        dyingSet.add(d.iid)
        stepDying.push(d.iid)
      }
    }
    dyingSet = dyingSet
    syncDisplay()
    // after the cross phase, drop each dying unit → its out:deathFx removal plays
    for (const id of stepDying) {
      setTimeout(() => {
        retained.delete(id)
        dyingSet.delete(id)
        dyingSet = dyingSet
        syncDisplay()
      }, CROSS_MS)
    }
    // clear the transient slide/flash after the hold
    setTimeout(() => {
      slideMap = new Map()
      flashSet = new Set()
    }, HOLD_MS)
  }

  // run the director once per step (fxToken bump)
  $: if (fxToken !== lastToken) {
    lastToken = fxToken
    onStep()
  }
  // keep the display synced to the view while fully idle (initial render, resync)
  $: if (view && retained.size === 0 && slideMap.size === 0) {
    displayMe = view.me.board
    displayOp = view.op.board
  }

  // --- drag-to-aim state (unchanged from Slice B) ---
  type Drag = { kind: 'attack' | 'use'; src: number; from: { x: number; y: number } }
  let drag: Drag | null = null
  let cursor = { x: 0, y: 0 }
  let snapId: number | 'face' | null = null

  function legalIdsFor(d: Drag): number[] {
    return d.kind === 'attack' ? attackTargets(legal, d.src) : itemTargets(legal, d.src)
  }
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
    e.preventDefault()
    drag = { kind: 'attack', src: c.iid, from: centerOf(e.currentTarget as HTMLElement) }
    cursor = { x: e.clientX, y: e.clientY }
    snapId = null
  }
  function downHand(e: MouseEvent, c: CardState) {
    if (!interactive) return
    const ts = itemTargets(legal, c.iid)
    if (ts.length === 0) return
    if (ts.length === 1 && ts[0] === -1) return
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
    if (sid === null) return
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
  $: lineTo = drag
    ? snapId !== null && anchors.get(snapId)
      ? centerOf(anchors.get(snapId) as HTMLElement)
      : cursor
    : null

  function isPlayable(c: CardState): boolean {
    return interactive && (canSummon(legal, c.iid) || itemTargets(legal, c.iid).length > 0)
  }
  const slideX = (iid: number) => slideMap.get(iid)?.dx ?? 0
  const slideY = (iid: number) => slideMap.get(iid)?.dy ?? 0
  // a minion that actually lost HP this step (combat: both attacker and defender)
  const tookDamage = (seat: number, iid: number) =>
    splashes.some((s) => s.seat === seat && s.target === iid && s.amount > 0)

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
  <!-- the opponent's whole player panel is the targetable face (drag a unit onto it) -->
  <div
    class="faceplate op"
    class:legaltarget={legalKeys.has('face')}
    class:snapped={snapId === 'face'}
    class:flashing={flashSet.has('face')}
    use:anchor={'face'}
    title="opponent — drag a unit here to attack">
    <Player player={opPlayer} name="AI" seat={opSeat as 0 | 1} active={false} {fx} {fxToken} />
  </div>

  <div class="hand backs">
    {#each oppBacks as _b, i (i)}<CardView card={back} faceUp={false} />{/each}
  </div>

  <div class="field top">
    {#each displayOp as c (c.iid)}
      <button class="slot" class:legaltarget={legalKeys.has(c.iid)} class:snapped={snapId === c.iid}
        use:anchor={c.iid} in:spring out:deathFx>
        <MinionView card={c} facing="down"
          slideX={slideX(c.iid)} slideY={slideY(c.iid)}
          flash={flashSet.has(c.iid)} hit={tookDamage(opSeat, c.iid)}
          dying={dyingSet.has(c.iid)} dmgDelay
          damage={cardDamage(splashes, opSeat, c.iid)} {fxToken} />
      </button>
    {/each}
  </div>

  <hr />

  <div class="field bottom">
    {#each displayMe as c (c.iid)}
      <button class="slot" class:legaltarget={legalKeys.has(c.iid)} class:snapped={snapId === c.iid}
        class:armed={drag?.src === c.iid} use:anchor={c.iid} in:spring out:deathFx
        on:mousedown={(e) => startAttack(e, c)}>
        <MinionView card={c} facing="up" dim={c.can_attack === false}
          slideX={slideX(c.iid)} slideY={slideY(c.iid)}
          flash={flashSet.has(c.iid)} hit={tookDamage(meSeat, c.iid)}
          dying={dyingSet.has(c.iid)} dmgDelay
          damage={cardDamage(splashes, meSeat, c.iid)} {fxToken} />
      </button>
    {/each}
  </div>

  <div class="hand mine" use:dock={{ enabled: interactive && !drag, target: '.card' }}>
    {#each view.me.hand as c (c.iid)}
      <button class="slot" class:playable={isPlayable(c)} class:armed={drag?.src === c.iid}
        in:dealIn on:mousedown={(e) => downHand(e, c)} on:click={() => clickHand(c)}>
        <CardView card={c} showAuras={false} />
      </button>
    {/each}
  </div>

  <!-- the human's own player panel is the bottom face (the AI's attack target) -->
  <div class="faceplate me" use:anchor={'face-me'}>
    <Player player={mePlayer} name="You" seat={meSeat as 0 | 1} active={true} {fx} {fxToken} />
  </div>

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
  /* perspective so the dealIn rotateY flip reads in 3D */
  .hand.mine { perspective: 900px; }
  .hand.backs { opacity: 0.85; }
  .slot { background: none; border: 2px solid transparent; border-radius: 8px; padding: 2px;
    cursor: pointer; transition: transform 0.12s ease, box-shadow 0.12s ease; }
  .slot:hover { border-color: #4a4f6a; }
  .slot.playable { border-color: #4fd97a; box-shadow: 0 0 9px rgba(79, 217, 122, 0.45); }
  .slot.armed { border-color: #ffd23d; box-shadow: 0 0 9px rgba(255, 210, 61, 0.5); }
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
  /* a player panel acting as the face hit-area */
  .faceplate { border: 2px solid transparent; border-radius: 8px; padding: 2px 6px;
    transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease; }
  /* a legal target during a drag, brighter when the line is snapped to it */
  .faceplate.legaltarget { background: rgba(90, 169, 255, 0.18); border-color: #5aa9ff; }
  .faceplate.snapped { background: rgba(255, 93, 93, 0.28); border-color: #ff5d5d;
    box-shadow: 0 0 16px rgba(255, 93, 93, 0.55); }
  /* cast flash on the face — reuse the existing brightness/scale pulse */
  .faceplate.flashing { animation: locma-cast 250ms ease-out; }
</style>
