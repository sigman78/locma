<script lang="ts">
  import type { CardMeta } from '../../lib/api'
  import type { AbilityInfo } from '../../lib/abilities'
  import { stripItemPreface, creatureSpecial } from '../../lib/cards'

  export let name: string
  export let meta: CardMeta | undefined = undefined
  export let baseAbil: AbilityInfo[] = []
  export let typeLabel = ''
  export let tip: 'above' | 'below' = 'above'
  export let baseAtk = 0
  export let baseDef = 0

  // items: atk/def stats + keyword pills carry no meaning, so hide them for spell cards
  $: isItem = !!meta && meta.type.startsWith('item')
  // bottom text = the card's special only: items drop the colour preface; creatures drop the
  // "X/Y Creature." preface + bare keyword sentences (shown above), leaving "Summon: ...".
  $: desc = !meta ? '' : isItem ? stripItemPreface(meta.description) : creatureSpecial(meta.description)
</script>

<div class="tooltip" class:tip-above={tip === 'above'} class:tip-below={tip === 'below'}>
  <div class="tt-head">
    <span class="tt-name">{meta?.name ?? name}</span>
    {#if meta}<span class="tt-cost">◆ {meta.cost}</span>{/if}
  </div>
  {#if meta}<div class="tt-type">{typeLabel}</div>{/if}
  <!-- creatures only: spell cards carry no meaningful atk/def stats -->
  {#if !isItem}
    <!-- tooltip mirrors the printed card (base stats), not the in-play buffed state -->
    <div class="tt-stats">
      <span class="atk">⚔ {baseAtk}</span>
      <span class="def">🛡 {baseDef}</span>
    </div>
  {/if}
  {#if baseAbil.length && !isItem}
    <div class="tt-keys">
      {#each baseAbil as a}
        <div class="tt-key"><span class="chip" style={`border-color:${a.color}`}>{a.emoji}</span> {a.name}</div>
      {/each}
    </div>
  {/if}
  {#if desc}<div class="tt-desc">{desc}</div>{/if}
</div>

<style>
  /* hover detail tooltip — centred above (or below) the card, never over a horizontal neighbour */
  .tooltip { position: absolute; left: 50%; z-index: 100;
    width: 220px; padding: 8px 10px; border-radius: 8px;
    background: #0d0f16; border: 1px solid #3a3f55;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6); color: #ddd;
    font-size: 12px; line-height: 1.4; text-align: left;
    opacity: 0; visibility: hidden; transform: translateX(-50%) translateY(4px);
    transition: opacity 0.12s ease, transform 0.12s ease; pointer-events: none; }
  .tooltip.tip-above { bottom: calc(100% + 8px); }
  .tooltip.tip-below { top: calc(100% + 8px); }
  .tt-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .tt-name { font-weight: 700; font-size: 13px; color: #fff; }
  .tt-cost { color: #6bb8ff; font-weight: 700; white-space: nowrap; }
  .tt-type { color: #99a; text-transform: capitalize; font-size: 11px; margin-top: 1px; }
  .tt-stats { display: flex; gap: 14px; margin: 5px 0; font-weight: 700; }
  .atk { color: #ffcc55; } .def { color: #66ccff; }
  .tt-keys { display: flex; flex-direction: column; gap: 3px; margin: 5px 0;
    border-top: 1px solid #2a2f42; padding-top: 5px; }
  .tt-key { display: flex; align-items: center; gap: 6px; }
  .tt-desc { margin-top: 5px; color: #bcbcc8; border-top: 1px solid #2a2f42; padding-top: 5px; }
  .chip { display: inline-block; min-width: 20px; text-align: center;
    font-size: 13px; line-height: 18px; border-radius: 4px; padding: 0 2px;
    background: rgba(8, 8, 12, 0.8); border: 1.5px solid #888; }
</style>
