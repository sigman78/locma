<!-- web/src/components/ReplayViewer/Hand.svelte -->
<script lang="ts">
  import type { CardState } from '../../lib/replay'
  import CardView from './CardView.svelte'
  export let cards: CardState[] = []
  export let faceUp = true
</script>

<div class="hand">
  {#each cards as c (c.iid)}<CardView card={c} {faceUp} />{/each}
</div>

<style>
  .hand { display: flex; gap: var(--gap, 8px); justify-content: flex-start; flex-wrap: nowrap;
    padding: 6px; background: #101015; border-radius: 6px;
    width: calc(var(--hand-cols, 8) * var(--card-w, 108px)
      + (var(--hand-cols, 8) - 1) * var(--gap, 8px) + 12px);
    min-height: calc(var(--card-h, 150px) + 12px); }
</style>
