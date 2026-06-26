<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import { card as cardMeta } from '../../lib/cards'
  import type { CardState } from '../../lib/replay'
  import type { DraftPending } from '../../lib/play'
  import CardView from '../ReplayViewer/CardView.svelte'
  import DeckStrip from './DeckStrip.svelte'
  import ManaCurve from './ManaCurve.svelte'

  export let pending: DraftPending
  // when the draft is finished we stay on this same view, but the card picker is
  // replaced by a Play button and the deck shows the full drafted list (doneCardIds).
  export let done = false
  export let doneCardIds: number[] = []
  const dispatch = createEventDispatcher<{ pick: number; auto: void; play: void }>()

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
  $: deckIds = done ? doneCardIds : pending.my_cards
</script>

<div class="draft">
  {#if done}
    <h2>Draft complete — {deckIds.length} cards</h2>
  {:else}
    <h2>Draft — round {pending.round + 1} / {pending.total}</h2>
  {/if}

  {#if !done}
    <div class="row">
      {#each cards as c, i (i)}
        <button class="pick" on:click={() => dispatch('pick', i)}>
          <CardView card={c} tipDir="below" />
        </button>
      {/each}
    </div>
  {/if}

  {#if deckIds.length > 0}
    <div class="deck-section">
      <ManaCurve cardIds={deckIds} />
      <DeckStrip cardIds={deckIds} label="Your deck" />
    </div>
  {/if}

  {#if done}
    <button class="play-btn" on:click={() => dispatch('play')}>Play ▶</button>
  {:else}
    <div class="foot">
      <span class="count">Drafted: {pending.my_picks} / {pending.total}</span>
      <button class="auto" on:click={() => dispatch('auto')}>Pick rest for me</button>
    </div>
  {/if}
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
  .deck-section { display: flex; flex-direction: column; align-items: center; gap: 8px;
    width: 100%; max-width: 700px; }
  .play-btn { background: #2a2a44; color: #fff; border: 1px solid #4a4f6a; border-radius: 4px;
    padding: 10px 28px; cursor: pointer; font-weight: 600; font-size: 1.1rem; margin-top: 4px;
    transition: background 0.12s, border-color 0.12s; }
  .play-btn:hover { background: #363660; border-color: #6a70a8; }
</style>
