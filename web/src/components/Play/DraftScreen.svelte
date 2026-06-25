<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import { card as cardMeta } from '../../lib/cards'
  import type { CardState } from '../../lib/replay'
  import type { DraftPending } from '../../lib/play'
  import CardView from '../ReplayViewer/CardView.svelte'

  export let pending: DraftPending
  const dispatch = createEventDispatcher<{ pick: number }>()

  function toCard(cardId: number, i: number): CardState {
    const m = cardMeta(cardId)
    return {
      iid: -1 - i,
      card_id: cardId,
      atk: m?.attack ?? 0,
      def: m?.defense ?? 0,
      abilities: m?.abilities ?? '',
    }
  }

  $: cards = pending.triplet.map(toCard)
</script>

<div class="draft">
  <h2>Draft — round {pending.round + 1} / {pending.total}</h2>
  <div class="row">
    {#each cards as c, i (i)}
      <button class="pick" on:click={() => dispatch('pick', i)}>
        <CardView card={c} />
      </button>
    {/each}
  </div>
  <p class="count">Cards drafted: {pending.my_picks}</p>
</div>

<style>
  .draft { --card-w: 140px; --card-h: 195px; color: #ddd; }
  .row { display: flex; gap: 20px; padding: 16px 0; }
  .pick { background: none; border: 2px solid transparent; border-radius: 8px;
    padding: 4px; cursor: pointer; transition: border-color 0.12s, transform 0.12s; }
  .pick:hover { border-color: #ffd23d; transform: translateY(-4px); }
  .count { color: #aaa; }
</style>
