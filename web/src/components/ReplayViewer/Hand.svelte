<!-- web/src/components/ReplayViewer/Hand.svelte -->
<script lang="ts">
  import { flip } from 'svelte/animate'
  import type { CardState } from '../../lib/replay'
  import { animate as fxWindow } from '../../lib/motion'
  import CardView from './CardView.svelte'
  export let cards: CardState[] = []
  export let faceUp = true
  export let active = false
  export let tipDir: 'above' | 'below' | null = null
  export let drawnIids: Set<number> = new Set() // hand cards to glow as freshly drawn
  export let fxToken = 0 // bump re-triggers the one-shot draw glow
</script>

<div class="hand" class:active>
  <!-- flip: neighbours glide to close the gap of a played card / make room for a
       draw — gated by the forward-step window so timeline seeks stay instant -->
  {#each cards as c (c.iid)}
    <div animate:flip={{ duration: $fxWindow ? 220 : 0 }}>
      <CardView card={c} {faceUp} {tipDir} drawn={drawnIids.has(c.iid)} {fxToken} />
    </div>
  {/each}
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
  .hand > div { flex: 0 0 auto; }
  /* faintly warm the active player's hand */
  .hand.active { background: #2a2a24; border-color: #4a4636; }
</style>
