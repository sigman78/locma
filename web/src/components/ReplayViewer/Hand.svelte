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

<style>.hand { display: flex; gap: 4px; }</style>
