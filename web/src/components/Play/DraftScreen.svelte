<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import { card as cardMeta } from '../../lib/cards'
  import { digitIndex, isTypingTarget } from '../../lib/keys'
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
  // keyboard only fires while the Play tab is visible (tabs stay mounted hidden)
  export let active = true
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

  // Keyboard (draft only): 1/2/3 pick a card, A auto-picks the rest. Starting
  // the battle stays a deliberate click — no Enter-to-play.
  function onKey(e: KeyboardEvent) {
    if (!active || done || e.altKey || e.ctrlKey || e.metaKey || isTypingTarget(e.target)) return
    const idx = digitIndex(e.key, cards.length)
    if (idx !== null) { e.preventDefault(); dispatch('pick', idx) }
    else if (e.key.toLowerCase() === 'a') { e.preventDefault(); dispatch('auto') }
  }
</script>

<svelte:window on:keydown={onKey} />

<div class="draft">
  {#if done}
    <h2>Draft complete — {deckIds.length} cards</h2>
  {:else}
    <h2>Draft — round {pending.round + 1} / {pending.total}</h2>
  {/if}

  {#if !done}
    <div class="row">
      {#each cards as c, i (i)}
        <button class="pick" title={`Pick (press ${i + 1})`} on:click={() => dispatch('pick', i)}>
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
      <button class="auto" title="Auto-pick the rest (press A)" on:click={() => dispatch('auto')}>
        Pick rest for me <span class="keycap">A</span>
      </button>
    </div>
  {/if}
</div>

<style>
  .draft { --card-w: 140px; --card-h: 195px; color: #ddd;
    display: flex; flex-direction: column; align-items: center; gap: 12px; padding-top: 24px; }
  h2 { margin: 0; }
  .row { display: flex; gap: 20px; padding: 16px 0; justify-content: center; }
  .pick { position: relative; background: none; border: 2px solid transparent; border-radius: 8px;
    padding: 4px; cursor: pointer;
    transition: border-color 0.14s, transform 0.14s ease, box-shadow 0.14s ease; }
  .keycap { font-size: 11px; font-weight: 600; color: #cbd0ec; background: #0e0e14;
    border: 1px solid #4a4f6a; border-radius: 3px; padding: 0 5px; margin-left: 4px; }
  /* lift the hovered card (and its tooltip) above its row neighbours */
  .pick:hover { border-color: #ffd23d; transform: translateY(-8px) scale(1.04);
    position: relative; z-index: 50; box-shadow: 0 14px 26px rgba(0, 0, 0, 0.5),
    0 0 12px rgba(255, 210, 61, 0.25); }
  .pick:active { transform: translateY(-4px) scale(1.0); }
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
