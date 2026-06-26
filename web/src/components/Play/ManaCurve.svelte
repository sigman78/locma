<script lang="ts">
  import { card as cardMeta } from '../../lib/cards'

  export let cardIds: number[]

  const LABELS = ['0', '1', '2', '3', '4', '5', '6', '7+']
  const BAR_H = 48 // max bar height in px

  $: counts = (() => {
    const c = [0, 0, 0, 0, 0, 0, 0, 0]
    for (const id of cardIds) {
      const cost = cardMeta(id)?.cost ?? 0
      c[Math.min(cost, 7)]++
    }
    return c
  })()

  $: maxCount = Math.max(...counts, 1) // avoid divide-by-zero
</script>

<div class="mana-curve">
  <span class="title">mana</span>
  <div class="bars">
    {#each counts as n, i}
      <div class="bucket">
        <span class="count" class:zero={n === 0}>{n === 0 ? '' : n}</span>
        <div class="bar" style={`height:${Math.max(Math.round((n / maxCount) * BAR_H), 2)}px`}></div>
        <span class="cost-label">{LABELS[i]}</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .mana-curve { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .title { color: #556; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .bars { display: flex; align-items: flex-end; gap: 5px; }
  .bucket { display: flex; flex-direction: column; align-items: center; gap: 2px; }
  .count { font-size: 0.65rem; color: #aaa; min-height: 12px; line-height: 12px; }
  .count.zero { visibility: hidden; }
  .bar { width: 18px; background: #5aa9ff; border-radius: 2px 2px 0 0;
    transition: height 0.15s ease; }
  .cost-label { font-size: 0.65rem; color: #556; }
</style>
