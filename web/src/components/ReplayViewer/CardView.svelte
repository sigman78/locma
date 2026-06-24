<script lang="ts">
  import type { CardState } from '../../lib/replay'
  import { artUrl, cardName } from '../../lib/cards'
  import { abilityList, hasAura } from '../../lib/abilities'
  import { restartAnim } from '../../lib/motion'

  export let card: CardState
  export let faceUp = true
  export let lunge: 'up' | 'down' | null = null
  export let damage: number | null = null
  export let fxToken = 0

  let imgOk = true
  $: name = cardName(card.card_id)
  $: abil = abilityList(card.abilities)
  $: guard = hasAura(card.abilities, 'G')
  $: ward = hasAura(card.abilities, 'W')
  $: lethal = hasAura(card.abilities, 'L')
  $: lungeCls = lunge ? `lunge-${lunge}` : null
</script>

{#if !faceUp}
  <div class="card back">🂠</div>
{:else}
  <div
    class="card"
    class:guard
    class:ward
    class:lethal
    class:attacked={card.has_attacked}
    use:restartAnim={{ cls: lungeCls, token: fxToken }}
  >
    {#if imgOk}
      <img src={artUrl(card.card_id)} alt={name} on:error={() => (imgOk = false)} />
    {:else}
      <div class="placeholder"><span class="nm">{name}</span></div>
    {/if}
    {#if lethal}<div class="tint"></div>{/if}
    <div class="stats"><span class="atk">{card.atk}</span><span class="def">{card.def}</span></div>
    <div class="abil">
      {#each abil as a}
        <span class="chip" style={`background:${a.color}`} title={a.name}>{a.letter}</span>
      {/each}
    </div>
    {#key fxToken}
      {#if damage != null}<div class="locma-dmg">-{damage}</div>{/if}
    {/key}
  </div>
{/if}

<style>
  .card { position: relative; width: var(--card-w, 108px); height: var(--card-h, 150px);
    border-radius: 6px; overflow: hidden; background: #1c1c22; border: 1px solid #333; }
  .card img { width: 100%; height: 100%; object-fit: cover; }
  .back { display: grid; place-items: center; font-size: 40px; color: #557;
    width: var(--card-w, 108px); height: var(--card-h, 150px);
    border-radius: 6px; background: #1c1c22; border: 1px solid #333; }
  .placeholder { display: grid; place-items: center; height: 100%; padding: 4px;
    text-align: center; font-size: 13px; color: #ddd; }
  /* auras (compose on different visual channels) */
  .card.guard { border: 2px solid #5aa9ff; box-shadow: inset 0 0 10px rgba(90,169,255,0.55); }
  .card.ward { box-shadow: 0 0 0 2px #7fe7ff, 0 0 12px 2px rgba(127,231,255,0.7); }
  .card.guard.ward { box-shadow: inset 0 0 10px rgba(90,169,255,0.55),
    0 0 0 2px #7fe7ff, 0 0 12px 2px rgba(127,231,255,0.7); }
  .tint { position: absolute; inset: 0; pointer-events: none;
    box-shadow: inset 0 0 18px 4px rgba(79,217,122,0.55);
    border: 1px solid rgba(79,217,122,0.6); border-radius: 6px; }
  .card.attacked { filter: saturate(0.6); }
  .stats { position: absolute; bottom: 0; left: 0; right: 0; display: flex;
    justify-content: space-between; padding: 3px 6px; font-weight: 700;
    background: rgba(0,0,0,0.6); font-size: 16px; }
  .atk { color: #ffcc55; } .def { color: #66ccff; }
  .abil { position: absolute; top: 3px; right: 3px; display: flex; flex-wrap: wrap;
    gap: 2px; max-width: 60%; justify-content: flex-end; }
  .chip { font-size: 11px; font-weight: 700; color: #0c0c10; border-radius: 3px;
    padding: 0 4px; line-height: 16px; text-shadow: 0 1px 0 rgba(255,255,255,0.3); }
</style>
