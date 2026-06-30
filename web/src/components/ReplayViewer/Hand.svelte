<!-- web/src/components/ReplayViewer/Hand.svelte -->
<script lang="ts">
  import type { CardState } from '../../lib/replay'
  import CardView from './CardView.svelte'
  export let cards: CardState[] = []
  export let faceUp = true
  export let active = false
  export let tipDir: 'above' | 'below' | null = null
  export let drawnIids: Set<number> = new Set() // hand cards to glow as freshly drawn
  export let fxToken = 0 // bump re-triggers the one-shot draw glow
</script>

<div class="hand" class:active>
  {#each cards as c (c.iid)}<CardView card={c} {faceUp} {tipDir} drawn={drawnIids.has(c.iid)} {fxToken} />{/each}
</div>

<style>
  .hand { display: flex; gap: var(--gap, 8px); justify-content: center; align-items: center;
    flex-wrap: nowrap; padding: 8px; border-radius: 8px;
    background: #20212b;
    border: 1px solid #313445;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    width: calc(var(--hand-cols, 8) * var(--card-w, 108px)
      + (var(--hand-cols, 8) - 1) * var(--gap, 8px) + 16px);
    min-height: calc(var(--card-h, 150px) + 16px); transition: background 0.2s, border-color 0.2s; }
  /* faintly warm the active player's hand */
  .hand.active { background: #2a2a24; border-color: #4a4636; }
</style>
