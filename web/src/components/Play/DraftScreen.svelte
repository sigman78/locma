<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import { card as cardMeta } from '../../lib/cards'
  import type { CardState } from '../../lib/replay'
  import type { DraftPending } from '../../lib/play'
  import CardView from '../ReplayViewer/CardView.svelte'

  export let pending: DraftPending
  const dispatch = createEventDispatcher<{ pick: number; auto: void }>()

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

  function cost(c: CardState): number {
    return cardMeta(c.card_id)?.cost ?? 0
  }

  $: cards = pending.triplet.map(toCard)
  $: drafted = (pending.my_cards ?? [])
    .map((id, i) => toCard(id, i))
    .sort((a, b) => (cost(a) - cost(b)) || (a.atk - b.atk))
</script>

<div class="draft">
  <h2>Draft — round {pending.round + 1} / {pending.total}</h2>
  <div class="row">
    {#each cards as c, i (i)}
      <button class="pick" on:click={() => dispatch('pick', i)}>
        <CardView card={c} tipDir="below" />
      </button>
    {/each}
  </div>
  {#if drafted.length > 0}
    <div class="strip-wrap">
      <span class="strip-label">Your deck ({drafted.length})</span>
      <div class="strip">
        {#each drafted as c, i (c.card_id + '-' + i)}
          <div class="strip-card">
            <CardView card={c} tipDir="above" />
          </div>
        {/each}
      </div>
    </div>
  {/if}
  <div class="foot">
    <span class="count">Drafted: {pending.my_picks} / {pending.total}</span>
    <button class="auto" on:click={() => dispatch('auto')}>Pick rest for me</button>
  </div>
</div>

<style>
  .draft { --card-w: 140px; --card-h: 195px; color: #ddd;
    display: flex; flex-direction: column; align-items: center; gap: 12px; padding-top: 24px; }
  h2 { margin: 0; }
  .row { display: flex; gap: 20px; padding: 16px 0; justify-content: center; }
  .pick { background: none; border: 2px solid transparent; border-radius: 8px;
    padding: 4px; cursor: pointer; transition: border-color 0.12s, transform 0.12s; }
  /* lift the hovered card (and its tooltip) above its row neighbours */
  .pick:hover { border-color: #ffd23d; transform: translateY(-4px); position: relative; z-index: 50; }
  .foot { display: flex; align-items: center; gap: 16px; }
  .count { color: #aaa; }
  .auto { background: #23232b; color: #ddd; border: 1px solid #4a4f6a; border-radius: 4px;
    padding: 6px 14px; cursor: pointer; font-weight: 600; }
  .auto:hover { background: #2a2a44; }
  /* drafted-cards strip */
  .strip-wrap { display: flex; flex-direction: column; align-items: flex-start;
    width: 100%; max-width: 700px; gap: 4px; }
  .strip-label { color: #888; font-size: 0.8rem; }
  .strip { --card-w: 80px; --card-h: 112px;
    display: flex; flex-wrap: wrap; padding-right: 27px; }
  /* position: relative + z-index: auto keeps CardView's :hover z-index: 50 untrapped */
  .strip-card { position: relative; margin-right: -27px; margin-bottom: 8px; }
</style>
