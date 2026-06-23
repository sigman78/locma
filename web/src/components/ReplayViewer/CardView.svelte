<script lang="ts">
  import type { CardState } from '../../lib/replay'
  import { artUrl, cardName } from '../../lib/cards'

  export let card: CardState
  export let faceUp = true

  let imgOk = true
  $: name = cardName(card.card_id)
  $: abilities = (card.abilities ?? '------').split('').filter((c) => c !== '-')
</script>

{#if !faceUp}
  <div class="card back">🂠</div>
{:else}
  <div class="card">
    {#if imgOk}
      <img src={artUrl(card.card_id)} alt={name} on:error={() => (imgOk = false)} />
    {:else}
      <div class="placeholder"><span class="nm">{name}</span></div>
    {/if}
    <div class="stats"><span class="atk">{card.atk}</span><span class="def">{card.def}</span></div>
    <div class="abil">{#each abilities as a}<span>{a}</span>{/each}</div>
  </div>
{/if}

<style>
  .card { position: relative; width: 72px; height: 100px; border-radius: 6px;
    overflow: hidden; background: #1c1c22; border: 1px solid #333; }
  .card img { width: 100%; height: 100%; object-fit: cover; }
  .back { display: grid; place-items: center; font-size: 28px; color: #557; }
  .placeholder { display: grid; place-items: center; height: 100%; padding: 4px;
    text-align: center; font-size: 10px; color: #ddd; }
  .stats { position: absolute; bottom: 0; left: 0; right: 0; display: flex;
    justify-content: space-between; padding: 2px 4px; font-weight: 700;
    background: rgba(0,0,0,0.55); font-size: 12px; }
  .atk { color: #ffcc55; } .def { color: #66ccff; }
  .abil { position: absolute; top: 2px; right: 2px; display: flex; gap: 1px;
    font-size: 8px; color: #fff; }
  .abil span { background: #4a3; border-radius: 2px; padding: 0 2px; }
</style>
