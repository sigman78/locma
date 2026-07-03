<!-- web/src/components/Play/BattleScreen.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import type { ActionDict, CardState, EventDict, PlayerState } from '../../lib/replay'
  import {
    attackTargets,
    breakthroughHit,
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
  import { card as cardMeta } from '../../lib/cards'
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
  // the common turn-1 case: nothing summonable, pass is the only move — say so
  $: passOnly = legal.length === 1 && legal[0].t === 'pass'
  $: meSeat = you
  $: opSeat = 1 - you
  $: interactive = !playing
  $: actSeat = liveStep ? liveStep.seat : you
  $: fx = computeFx(events, currentAction, actSeat)
  $: splashes = splashesFor(events)

  function send(a: ActionDict) {
    drag = null
    snapId = null
    overField = false
    handDrag = null
    dispatch('act', a)
  }

  // --- anchor registry: board slots + both faces register their DOM node.
  // 'face' = the opponent (top) face (also the drag target); 'face-me' = the
  // human's own (bottom) face, needed so an AI→human-face attack can slide down. ---
  type AnchorKey = number | 'face' | 'face-me' | 'myfield'
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
  type BtFly = {
    amount: number
    src: { cx: number; cy: number }
    dst: { cx: number; cy: number }
    key: number
  }
  let btFly: BtFly | null = null

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
    // Breakthrough cue: a minion attack that also overflowed onto the defender face
    // → animate a red number flying from the struck/dead blocker to the face.
    // Coords are captured NOW on the still-current DOM (the target may die this step).
    const bht = breakthroughHit(currentAction, splashes, actSeat)
    if (bht && currentAction?.t === 'attack') {
      const src = rectOfKey(currentAction.target)
      const faceKey: 'face' | 'face-me' = actSeat === you ? 'face' : 'face-me'
      const dst = rectOfKey(faceKey)
      if (src && dst) {
        const flyKey = fxToken
        // small delay (~80ms) so the projectile fires near the slide apex
        setTimeout(() => { btFly = { amount: bht.amount, src, dst, key: flyKey } }, 80)
        // clear AFTER the fade fully completes (80ms delay + 390ms animation-delay + 140ms fade ≈ 610ms)
        setTimeout(() => { if (btFly?.key === flyKey) btFly = null }, 640)
      }
    }
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
  // summons are NOT a Drag: the hand card itself is carried (see handDrag below)
  type Drag = { kind: 'attack' | 'use'; src: number; from: { x: number; y: number } }
  let drag: Drag | null = null
  let cursor = { x: 0, y: 0 }
  let snapId: number | 'face' | null = null
  let overField = false // cursor is over the own battlefield during a summon drag

  function legalIdsFor(d: Drag): number[] {
    if (d.kind === 'attack') return attackTargets(legal, d.src)
    if (d.kind === 'use') return itemTargets(legal, d.src)
    return [] // summon drops on the field, not a specific slot target
  }

  function pointInField(p: { x: number; y: number }): boolean {
    const el = anchors.get('myfield')
    if (!el) return false
    const r = el.getBoundingClientRect()
    return p.x >= r.left && p.x <= r.right && p.y >= r.top && p.y <= r.bottom
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

  // illegal-action feedback: quick head-shake on the clicked card
  let jiggleIid: number | null = null
  let jiggleTimer: ReturnType<typeof setTimeout> | null = null
  function jiggle(iid: number) {
    // synchronous set (no rAF: throttled in occluded windows); a re-click during
    // the 400ms window extends the timer instead of restarting the animation
    jiggleIid = iid
    if (jiggleTimer) clearTimeout(jiggleTimer)
    jiggleTimer = setTimeout(() => (jiggleIid = null), 400)
  }

  function startAttack(e: MouseEvent, c: CardState) {
    if (!interactive) return
    if (attackTargets(legal, c.iid).length === 0) {
      jiggle(c.iid) // summoning-sick / already attacked / no targets
      return
    }
    e.preventDefault()
    drag = { kind: 'attack', src: c.iid, from: centerOf(e.currentTarget as HTMLElement) }
    cursor = { x: e.clientX, y: e.clientY }
    snapId = null
  }
  // CREATURE drag: the hand card itself is carried (no pointer-line arrow).
  // live=true (summonable): the card tracks the cursor 1:1 and may be dropped
  // on the own battlefield to summon — dropped anywhere else it springs back
  // to its hand slot. live=false (unplayable: mana / board space): an elastic
  // tether — displacement saturates toward DEAD_MAX and pulling past
  // DEAD_BREAK snaps the card out of your grip. Spell cards keep the old
  // click / aim-drag behavior.
  const DEAD_MAX = 56
  const DEAD_BREAK = 150
  let handDrag: {
    iid: number
    live: boolean
    sx: number
    sy: number
    dx: number
    dy: number
    returning: boolean
  } | null = null
  let deadMoved = false // suppress the click-jiggle when the tether already gave feedback

  $: carrying = !!handDrag && handDrag.live && !handDrag.returning

  function startHandDrag(e: MouseEvent, iid: number, live: boolean) {
    e.preventDefault()
    deadMoved = false
    handDrag = { iid, live, sx: e.clientX, sy: e.clientY, dx: 0, dy: 0, returning: false }
    if (live) {
      cursor = { x: e.clientX, y: e.clientY }
      overField = false
    }
  }
  function handRelease() {
    if (!handDrag || handDrag.returning) return
    const iid = handDrag.iid
    handDrag = { ...handDrag, dx: 0, dy: 0, returning: true }
    overField = false
    setTimeout(() => {
      if (handDrag?.iid === iid && handDrag.returning) handDrag = null
    }, 340)
  }

  function downHand(e: MouseEvent, c: CardState) {
    if (!interactive) return
    // a summonable creature: carry it — drop on your battlefield to summon
    if (canSummon(legal, c.iid)) {
      startHandDrag(e, c.iid, true)
      return
    }
    // a creature that CAN'T be summoned (mana / board space): elastic tether
    if ((cardMeta(c.card_id)?.type ?? 'creature') === 'creature') {
      startHandDrag(e, c.iid, false)
      return
    }
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
    if (ts.length === 1 && ts[0] === -1) {
      send({ t: 'use', item: c.iid, target: -1 })
    } else if (ts.length === 0 && !deadMoved) {
      jiggle(c.iid) // plain click on an unplayable card (a real tether pull already said no)
    }
  }
  function onMove(e: MouseEvent) {
    if (handDrag && !handDrag.returning) {
      const rx = e.clientX - handDrag.sx
      const ry = e.clientY - handDrag.sy
      const dist = Math.hypot(rx, ry)
      if (dist > 8) deadMoved = true
      if (handDrag.live) {
        // free 1:1 carry; the own battlefield highlights as the drop zone
        handDrag = { ...handDrag, dx: rx, dy: ry }
        cursor = { x: e.clientX, y: e.clientY }
        overField = pointInField(cursor)
        return
      }
      if (dist > DEAD_BREAK) {
        handRelease() // pulled too far: the card escapes the grip
        return
      }
      // rubber-band: displacement saturates toward DEAD_MAX as the pull grows
      const f = dist > 0 ? (DEAD_MAX * (1 - Math.exp(-dist / DEAD_MAX))) / dist : 0
      handDrag = { ...handDrag, dx: rx * f, dy: ry * f }
      return
    }
    if (!drag) return
    cursor = { x: e.clientX, y: e.clientY }
    const best = nearestTarget(cursor.x, cursor.y, aimTargets(drag))
    snapId = best ? best.id : null
  }
  function onUp() {
    if (handDrag && !handDrag.returning) {
      if (handDrag.live && overField) {
        send({ t: 'summon', id: handDrag.iid }) // dropped on the own battlefield
      } else {
        handRelease() // dropped anywhere else: spring back to the hand
      }
      return
    }
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
      overField = false
      if (handDrag) handRelease()
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

  <div class="field bottom"
    class:summon-target={carrying}
    class:summon-over={carrying && overField}
    use:anchor={'myfield'}>
    {#each displayMe as c (c.iid)}
      <button class="slot" class:legaltarget={legalKeys.has(c.iid)} class:snapped={snapId === c.iid}
        class:armed={drag?.src === c.iid} class:jiggling={jiggleIid === c.iid}
        class:ready={interactive && !drag && attackTargets(legal, c.iid).length > 0}
        use:anchor={c.iid} in:spring out:deathFx
        on:mousedown={(e) => startAttack(e, c)}>
        <MinionView card={c} facing="up" dim={c.can_attack === false}
          slideX={slideX(c.iid)} slideY={slideY(c.iid)}
          flash={flashSet.has(c.iid)} hit={tookDamage(meSeat, c.iid)}
          dying={dyingSet.has(c.iid)} dmgDelay
          damage={cardDamage(splashes, meSeat, c.iid)} {fxToken} />
      </button>
    {/each}
  </div>

  <div class="hand mine" use:dock={{ enabled: interactive && !drag && !handDrag, target: '.card' }}>
    {#each view.me.hand as c (c.iid)}
      <button class="slot" class:playable={isPlayable(c)} class:armed={drag?.src === c.iid}
        class:jiggling={jiggleIid === c.iid}
        class:tethered={handDrag?.iid === c.iid && !handDrag.returning}
        class:carrying={handDrag?.iid === c.iid && handDrag.live && !handDrag.returning}
        class:tether-return={handDrag?.iid === c.iid && handDrag.returning}
        style={handDrag?.iid === c.iid
          ? `transform: translate(${handDrag.dx}px, ${handDrag.dy}px) ` +
            `rotate(${Math.max(-7, Math.min(7, handDrag.dx * 0.06))}deg)` +
            (handDrag.live && !handDrag.returning ? ' scale(1.06)' : '') + ';'
          : ''}
        in:dealIn on:mousedown={(e) => downHand(e, c)} on:click={() => clickHand(c)}>
        <CardView card={c} />
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
      {#if playing}AI is taking its turn…{:else if drag}Drag to a highlighted target — release to confirm, Esc to cancel.{:else if passOnly}No playable actions this turn — end your turn.{:else}Your turn — drag a unit to attack, click or drag a card to your field to summon, drag an item to its target, or end turn.{/if}
    </span>
    <button
      class="endturn"
      class:urge={passOnly && interactive}
      on:click={() => send({ t: 'pass' })}
      disabled={!interactive}>End Turn ⏭</button>
  </div>
</div>

{#if btFly}
  <!-- Breakthrough red blob flies from struck blocker → defender face (position:fixed, viewport coords).
       No number on the blob; the Player's own floating face-HP "-N" number shows normally. -->
  <div
    class="bt-fly"
    style="left:{btFly.src.cx}px; top:{btFly.src.cy}px; --dx:{btFly.dst.cx - btFly.src.cx}px; --dy:{btFly.dst.cy - btFly.src.cy}px"
  ></div>
{/if}

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
  /* drag-to-summon: highlight the own battlefield as the drop zone */
  .field.summon-target { outline: 2px dashed #5aa9ff; outline-offset: -3px;
    background: rgba(90, 169, 255, 0.08); }
  .field.summon-over { outline: 2px solid #5aa9ff; outline-offset: -3px;
    background: rgba(90, 169, 255, 0.2); box-shadow: inset 0 0 18px rgba(90, 169, 255, 0.45); }
  .hand { display: flex; gap: var(--gap); justify-content: center; align-items: center; padding: 6px;
    background: #20212b; border: 1px solid #313445; border-radius: 8px;
    width: calc(var(--hand-cols) * var(--card-w) + (var(--hand-cols) - 1) * var(--gap) + 16px);
    min-height: calc(var(--card-h) + 12px); }
  /* perspective so the dealIn rotateY flip reads in 3D */
  .hand.mine { perspective: 900px; }
  .hand.backs { opacity: 0.85; }
  /* The slot is a <button>: reset its UA defaults — both the border bevel AND the
     browser's default ~6px inline padding — so a slot's layout width == --card-w and
     all 8 hand cards fit the panel. Highlights use outline, which takes no layout space. */
  .slot { background: none; border: none; padding: 0; outline: 2px solid transparent;
    outline-offset: -2px; border-radius: 8px; cursor: pointer; position: relative;
    transition: transform 0.14s ease, box-shadow 0.14s ease, outline-color 0.14s ease; }
  .slot:hover { outline-color: #4a4f6a; }
  /* hover juice: hand cards lift toward you, board minions perk up */
  .hand.mine .slot:hover { transform: translateY(-10px) scale(1.05); z-index: 30;
    box-shadow: 0 12px 22px rgba(0, 0, 0, 0.5); }
  /* carried / tethered hand card: tight cursor tracking while held (the inline
     transform wins over :hover), springy overshoot on the way back */
  .slot.tethered { transition: transform 0.05s linear; z-index: 40;
    cursor: grabbing; }
  .slot.carrying { box-shadow: 0 16px 30px rgba(0, 0, 0, 0.55); }
  .slot.tether-return { transition: transform 320ms cubic-bezier(0.34, 1.56, 0.64, 1);
    z-index: 40; }
  .field .slot:hover { transform: translateY(-3px) scale(1.03); z-index: 30; }
  /* a minion with legal attack targets: soft amber ready-glow */
  .slot.ready { outline-color: rgba(255, 210, 61, 0.55);
    animation: ready-breathe 2.6s ease-in-out infinite; }
  .slot.playable { outline-color: #4fd97a; box-shadow: 0 0 9px rgba(79, 217, 122, 0.45);
    animation: playable-breathe 2.4s ease-in-out infinite; }
  .slot.armed { outline-color: #ffd23d; box-shadow: 0 0 9px rgba(255, 210, 61, 0.5);
    animation: none; }
  .slot.legaltarget { outline-color: #5aa9ff; box-shadow: 0 0 8px rgba(90, 169, 255, 0.5);
    animation: none; }
  .slot.snapped { outline-color: #ff5d5d; box-shadow: 0 0 12px rgba(255, 93, 93, 0.85);
    animation: none; }
  @keyframes playable-breathe {
    50% { box-shadow: 0 0 15px rgba(79, 217, 122, 0.75); }
  }
  @keyframes ready-breathe {
    50% { outline-color: rgba(255, 210, 61, 0.25); }
  }
  hr { width: 70%; border: none; border-top: 1px dashed #3a4a3c; margin: 2px 0; }
  /* always on screen: the board can be taller than the viewport inside the
     tabbed shell, and an invisible End Turn reads as a frozen game */
  .controls { display: flex; gap: 16px; align-items: center; margin-top: 4px;
    position: sticky; bottom: 0; z-index: 40; background: rgba(14, 14, 18, 0.92);
    padding: 8px 4px; border-top: 1px solid #23232b; }
  .endturn.urge { border-color: #3fbf66; box-shadow: 0 0 10px rgba(79, 217, 122, 0.35);
    animation: urge-pulse 1.6s ease-in-out infinite; }
  @keyframes urge-pulse {
    50% { box-shadow: 0 0 16px rgba(79, 217, 122, 0.6); }
  }
  .turnno { color: #ffd23d; font-weight: 700; font-size: 14px;
    background: rgba(255, 210, 61, 0.12); border: 1px solid #ffd23d55;
    border-radius: 10px; padding: 2px 10px; }
  .hint { color: #aaa; font-size: 14px; }
  .endturn { background: #2a2a44; color: #fff; border: 1px solid #4a4f6a;
    border-radius: 4px; padding: 8px 18px; cursor: pointer; font-weight: 600;
    transition: transform 0.12s ease, filter 0.12s ease; }
  .endturn:not(:disabled):hover { transform: translateY(-1px); filter: brightness(1.2); }
  .endturn:not(:disabled):active { transform: translateY(0) scale(0.97); }
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

  /* ---- Breakthrough flying-blob cue ---- */
  /* Pure red blob flies from struck blocker to defender face with back-overshoot spring.
     No number on the blob — the Player's own face-HP "-N" number shows normally. */
  .bt-fly {
    position: fixed;
    transform: translate(-50%, -50%);
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #ff4444;
    box-shadow: 0 0 10px rgba(255, 80, 80, 0.95), 0 0 24px rgba(255, 0, 0, 0.6);
    pointer-events: none;
    z-index: 999;
    /* springy move + fade-out after arrival */
    animation:
      bt-fly-move 300ms cubic-bezier(0.34, 1.56, 0.64, 1) both,
      bt-fly-out  140ms ease-in 390ms both;
    will-change: transform;
  }
  /* trailing glow blob — fades as the number moves, giving a brief afterimage streak */
  .bt-fly::before {
    content: '';
    position: absolute;
    left: 50%; top: 50%;
    width: 44px; height: 44px;
    margin: -22px 0 0 -22px;
    background: rgba(255, 60, 60, 0.5);
    border-radius: 50%;
    filter: blur(8px);
    animation: bt-trail 300ms ease-out both;
  }
  @keyframes bt-fly-move {
    from { transform: translate(-50%, -50%); }
    to   { transform: translate(calc(-50% + var(--dx)), calc(-50% + var(--dy))); }
  }
  @keyframes bt-fly-out {
    to { opacity: 0; }
  }
  @keyframes bt-trail {
    from { opacity: 0.85; transform: scale(1.8); }
    to   { opacity: 0;    transform: scale(0.3); }
  }
</style>
