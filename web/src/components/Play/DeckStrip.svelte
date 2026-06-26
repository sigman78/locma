<script lang="ts">
  import { card as cardMeta } from '../../lib/cards'
  import type { CardState } from '../../lib/replay'
  import CardView from '../ReplayViewer/CardView.svelte'

  export let cardIds: number[]
  export let label: string = ''
  export let cardW: number = 80

  $: cardH = Math.round(cardW * 1.4)

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

  $: sorted = cardIds
    .map((id, i) => toCard(id, i))
    .sort((a, b) => {
      const ca = cardMeta(a.card_id)?.cost ?? 0
      const cb = cardMeta(b.card_id)?.cost ?? 0
      return (ca - cb) || (a.atk - b.atk)
    })
</script>

{#if cardIds.length > 0}
  {#if label}
    <span class="strip-label">{label} ({cardIds.length})</span>
  {/if}
  <div class="strip" style={`--card-w:${cardW}px; --card-h:${cardH}px`}>
    {#each sorted as c, i (c.card_id + '-' + i)}
      <div class="strip-card">
        <CardView card={c} tipDir="above" />
      </div>
    {/each}
  </div>
{/if}

<style>
  .strip-label { color: #888; font-size: 0.8rem; display: block; margin-bottom: 4px; }
  .strip { display: flex; flex-wrap: wrap; padding-right: calc(var(--card-w, 80px) / 3); }
  /* position: relative + z-index: auto keeps CardView's :hover z-index: 50 untrapped */
  .strip-card { position: relative; margin-right: calc(var(--card-w, 80px) / -3); margin-bottom: 8px; }
</style>
