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
  } from '../../lib/play'
  import CardView from '../ReplayViewer/CardView.svelte'
  import Player from '../ReplayViewer/Player.svelte'

  export let pending: BattlePending
  export let you: number
  export let events: EventDict[] = []
  export let fxToken = 0

  const dispatch = createEventDispatcher<{ act: ActionDict }>()

  let selectedAttacker: number | null = null
  let selectedItem: number | null = null

  $: view = pending.view
  $: legal = pending.legal
  $: meSeat = you
  $: opSeat = 1 - you
  $: splashes = splashesFor(events)

  // reset transient selections whenever a new server state arrives
  $: if (pending) {
    selectedAttacker = null
    selectedItem = null
  }

  function send(a: ActionDict) {
    selectedAttacker = null
    selectedItem = null
    dispatch('act', a)
  }

  // cancel a pending attacker/item selection (re-click, Cancel button, or Esc)
  function cancel() {
    selectedAttacker = null
    selectedItem = null
  }
  function onKey(e: KeyboardEvent) {
    if (e.key === 'Escape') cancel()
  }
  $: selecting = selectedAttacker !== null || selectedItem !== null

  function clickHand(c: CardState) {
    if (selectedItem === c.iid) {
      selectedItem = null // re-clicking the chosen item cancels targeting
      return
    }
    if (canSummon(legal, c.iid)) {
      send({ t: 'summon', id: c.iid })
      return
    }
    const targets = itemTargets(legal, c.iid)
    if (targets.length === 0) return
    if (targets.length === 1 && targets[0] === -1) {
      send({ t: 'use', item: c.iid, target: -1 })
      return
    }
    selectedItem = c.iid // await a target click
  }

  function clickMyBoard(c: CardState) {
    if (selectedItem !== null && itemTargets(legal, selectedItem).includes(c.iid)) {
      send({ t: 'use', item: selectedItem, target: c.iid })
      return
    }
    if (attackTargets(legal, c.iid).length > 0) {
      selectedAttacker = selectedAttacker === c.iid ? null : c.iid
    }
  }

  function clickOpBoard(c: CardState) {
    if (selectedAttacker !== null && attackTargets(legal, selectedAttacker).includes(c.iid)) {
      send({ t: 'attack', a: selectedAttacker, target: c.iid })
      return
    }
    if (selectedItem !== null && itemTargets(legal, selectedItem).includes(c.iid)) {
      send({ t: 'use', item: selectedItem, target: c.iid })
    }
  }

  function clickOpFace() {
    if (selectedAttacker !== null && attackTargets(legal, selectedAttacker).includes(-1)) {
      send({ t: 'attack', a: selectedAttacker, target: -1 })
      return
    }
    if (selectedItem !== null && itemTargets(legal, selectedItem).includes(-1)) {
      send({ t: 'use', item: selectedItem, target: -1 })
    }
  }

  // PlayerState-shaped objects so the existing Player component renders the stat rows.
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

  // a face-down placeholder card for the opponent hand
  const back: CardState = { iid: -999, card_id: 0, atk: 0, def: 0, abilities: '' }
  $: oppBacks = new Array(view.op.hand_count).fill(back)
  $: fx = { lunge: null, cast: null, splashes }
</script>

<svelte:window on:keydown={onKey} />

<div class="battle">
  <Player player={opPlayer} name="AI" seat={opSeat as 0 | 1} active={false} {fx} {fxToken} />
  <button class="face op" on:click={clickOpFace} title="attack opponent">🎯 face</button>

  <div class="hand backs">
    {#each oppBacks as _b, i (i)}<CardView card={back} faceUp={false} />{/each}
  </div>

  <div class="field top">
    {#each view.op.board as c (c.iid)}
      <button class="slot" on:click={() => clickOpBoard(c)}>
        <CardView card={c} facing="down" damage={cardDamage(splashes, opSeat, c.iid)} {fxToken} />
      </button>
    {/each}
  </div>

  <hr />

  <div class="field bottom">
    {#each view.me.board as c (c.iid)}
      <button class="slot" class:selected={selectedAttacker === c.iid} on:click={() => clickMyBoard(c)}>
        <CardView card={c} facing="up" dim={c.can_attack === false}
          damage={cardDamage(splashes, meSeat, c.iid)} {fxToken} />
      </button>
    {/each}
  </div>

  <div class="hand mine">
    {#each view.me.hand as c (c.iid)}
      <button class="slot" class:selected={selectedItem === c.iid} on:click={() => clickHand(c)}>
        <CardView card={c} showAuras={false} />
      </button>
    {/each}
  </div>

  <Player player={mePlayer} name="You" seat={meSeat as 0 | 1} active={true} {fx} {fxToken} />

  <div class="controls">
    <span class="hint">
      {#if selectedAttacker !== null}Pick a target (or opponent face) — Esc to cancel.{:else if selectedItem !== null}Pick an item target — Esc to cancel.{:else}Your turn — summon, attack, or end turn.{/if}
    </span>
    {#if selecting}<button class="cancel" on:click={cancel}>✕ Cancel</button>{/if}
    <button class="endturn" on:click={() => send({ t: 'pass' })}>End Turn ⏭</button>
  </div>
</div>

<style>
  .battle { --card-w: 100px; --card-h: 140px; --gap: 8px;
    display: flex; flex-direction: column; gap: 8px; align-items: center;
    background: #15151b; border-radius: 8px; padding: 14px; color: #ddd; }
  .field { display: flex; gap: var(--gap); align-items: center; justify-content: center;
    min-height: calc(var(--card-h) + 12px); padding: 6px;
    background: rgba(255, 255, 255, 0.02); border-radius: 6px;
    width: calc(6 * var(--card-w) + 5 * var(--gap) + 16px); }
  .hand { display: flex; gap: var(--gap); justify-content: center; padding: 6px;
    background: #20212b; border: 1px solid #313445; border-radius: 8px; min-height: calc(var(--card-h) + 12px); }
  .hand.backs { opacity: 0.85; }
  .slot { background: none; border: 2px solid transparent; border-radius: 8px; padding: 2px; cursor: pointer; }
  .slot:hover { border-color: #4a4f6a; }
  .slot.selected { border-color: #ffd23d; }
  hr { width: 70%; border: none; border-top: 1px dashed #3a4a3c; margin: 2px 0; }
  .controls { display: flex; gap: 16px; align-items: center; margin-top: 4px; }
  .hint { color: #aaa; font-size: 14px; }
  .endturn { background: #2a2a44; color: #fff; border: 1px solid #4a4f6a;
    border-radius: 4px; padding: 8px 18px; cursor: pointer; font-weight: 600; }
  .cancel { background: #3a2330; color: #ffc4d6; border: 1px solid #6a3a4f;
    border-radius: 4px; padding: 8px 16px; cursor: pointer; font-weight: 600; }
  .cancel:hover { background: #4a2c3c; }
  .face.op { background: #2b1a1a; color: #ffb4b4; border: 1px solid #5a3a3a;
    border-radius: 4px; padding: 4px 12px; cursor: pointer; }
</style>
