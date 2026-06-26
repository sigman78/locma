<!-- web/src/components/Play/DraftComplete.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import DeckStrip from './DeckStrip.svelte'
  import ManaCurve from './ManaCurve.svelte'

  export let cardIds: number[]

  const dispatch = createEventDispatcher<{ play: void }>()
</script>

<div class="draft-complete">
  <h2>Draft complete — {cardIds.length} cards</h2>
  <ManaCurve {cardIds} />
  <DeckStrip {cardIds} cardW={104} />
  <button class="play-btn" on:click={() => dispatch('play')}>Play ▶</button>
</div>

<style>
  .draft-complete {
    color: #ddd;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    padding-top: 24px;
  }
  h2 {
    margin: 0;
  }
  .play-btn {
    background: #2a2a44;
    color: #fff;
    border: 1px solid #4a4f6a;
    border-radius: 4px;
    padding: 10px 28px;
    cursor: pointer;
    font-weight: 600;
    font-size: 1.1rem;
    margin-top: 8px;
    transition: background 0.12s, border-color 0.12s;
  }
  .play-btn:hover {
    background: #363660;
    border-color: #6a70a8;
  }
</style>
