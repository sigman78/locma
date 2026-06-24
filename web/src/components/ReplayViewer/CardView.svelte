<script lang="ts">
  import type { CardState } from '../../lib/replay'
  import { artUrl, cardName, card as cardMeta } from '../../lib/cards'
  import { abilityList, hasAura } from '../../lib/abilities'
  import { restartAnim } from '../../lib/motion'

  export let card: CardState
  export let faceUp = true
  export let lunge: 'up' | 'down' | null = null
  export let damage: number | null = null
  export let fxToken = 0
  export let dim = false
  export let showAuras = true
  export let facing: 'up' | 'down' | null = null // direction toward the opponent

  let imgOk = true
  $: name = cardName(card.card_id)
  $: meta = cardMeta(card.card_id)
  $: abil = abilityList(card.abilities)
  $: guard = hasAura(card.abilities, 'G')
  $: ward = hasAura(card.abilities, 'W')
  $: lethal = hasAura(card.abilities, 'L')
  $: lungeCls = lunge ? `lunge-${lunge}` : null

  // red / green / blue item typing (creatures have no item kind)
  const ITEM = {
    itemgreen: { color: '#4fd97a', emoji: '🟢', label: 'Green item' },
    itemred: { color: '#ff5d5d', emoji: '🔴', label: 'Red item' },
    itemblue: { color: '#5aa9ff', emoji: '🔵', label: 'Blue item' },
  } as const
  $: item = meta ? (ITEM as Record<string, { color: string; emoji: string; label: string }>)[meta.type] : undefined
  $: typeLabel = meta ? (item ? `${item.emoji} ${item.label}` : 'Creature') : ''
</script>

{#if !faceUp}
  <div class="cardwrap"><div class="card back">🂠</div></div>
{:else}
  <div class="cardwrap">
    <div
      class="card"
      class:guard={showAuras && guard}
      class:ward={showAuras && ward}
      class:lethal={showAuras && lethal}
      class:face-up={facing === 'up'}
      class:face-down={facing === 'down'}
      class:attacking={!!lunge}
      class:attacked={card.has_attacked}
      class:dim
      use:restartAnim={{ cls: lungeCls, token: fxToken }}
    >
      {#if imgOk}
        <img src={artUrl(card.card_id)} alt={name} on:error={() => (imgOk = false)} />
      {:else}
        <div class="placeholder"><span class="nm">{name}</span></div>
      {/if}
      {#if showAuras && ward}<div class="ward-tint"></div>{/if}
      <div class="stats">
        <span class="atk">{card.atk}</span>
        {#if item}<span class="item-dot" style={`background:${item.color}`} title={item.label}></span>{/if}
        <span class="def">{card.def}</span>
      </div>
      <div class="abil">
        {#each abil as a}
          <span class="chip" style={`border-color:${a.color}`} title={a.name}>{a.emoji}</span>
        {/each}
      </div>
      {#key fxToken}
        {#if damage != null}<div class="locma-dmg">-{damage}</div>{/if}
      {/key}
    </div>

    <div class="tooltip">
      <div class="tt-head">
        <span class="tt-name">{meta?.name ?? name}</span>
        {#if meta}<span class="tt-cost">◆ {meta.cost}</span>{/if}
      </div>
      {#if meta}<div class="tt-type">{typeLabel}</div>{/if}
      <div class="tt-stats"><span class="atk">⚔ {card.atk}</span><span class="def">🛡 {card.def}</span></div>
      {#if abil.length}
        <div class="tt-keys">
          {#each abil as a}
            <div class="tt-key"><span class="chip" style={`border-color:${a.color}`}>{a.emoji}</span> {a.name}</div>
          {/each}
        </div>
      {/if}
      {#if meta?.description}<div class="tt-desc">{meta.description}</div>{/if}
    </div>
  </div>
{/if}

<style>
  .cardwrap { position: relative; width: var(--card-w, 108px); height: var(--card-h, 150px); }
  .cardwrap:hover { z-index: 40; }
  .card { position: relative; width: 100%; height: 100%;
    border-radius: 6px; overflow: hidden; border: 1px solid #333;
    background-color: #1c1c22;
    background-image:
      repeating-linear-gradient(45deg, rgba(255, 255, 255, 0.04) 0 1px, transparent 1px 6px),
      repeating-linear-gradient(-45deg, rgba(255, 255, 255, 0.025) 0 1px, transparent 1px 6px); }
  .card img { width: 100%; height: 100%; object-fit: cover; }
  .back { display: grid; place-items: center; font-size: 40px; color: #557; }
  .placeholder { display: grid; place-items: center; height: 100%; padding: 4px;
    text-align: center; font-size: 13px; color: #ddd; }
  /* auras (battlefield only; each on an independent visual channel) */
  /* Lethal — green outline */
  .card.lethal { outline: 2px solid #4fd97a; outline-offset: 0; }
  /* Guard — bold white edge on the opponent-facing side ("shield wall") */
  .card.guard.face-up { border-top: 4px solid #fff;
    box-shadow: 0 -2px 7px rgba(255, 255, 255, 0.5); }
  .card.guard.face-down { border-bottom: 4px solid #fff;
    box-shadow: 0 2px 7px rgba(255, 255, 255, 0.5); }
  /* Ward — inner bubble: light-blue tint over the sprite + inner glow */
  .ward-tint { position: absolute; inset: 0; pointer-events: none; border-radius: 6px;
    background: rgba(127, 231, 255, 0.12);
    box-shadow: inset 0 0 16px 3px rgba(127, 231, 255, 0.7); }
  .card.attacked { filter: saturate(0.6); }
  /* summoning-sick / inactive dim — on the card only, so the tooltip stays opaque */
  .card.dim { opacity: 0.5; }
  /* attacker highlight — declared after .attacked so it wins the filter */
  .card.attacking { outline: 3px solid #ffd23d; outline-offset: 2px;
    filter: brightness(1.18); opacity: 1; z-index: 4; }
  .stats { position: absolute; bottom: 0; left: 0; right: 0; display: flex;
    justify-content: space-between; padding: 3px 6px; font-weight: 700;
    background: rgba(0,0,0,0.6); font-size: 16px; }
  .atk { color: #ffcc55; } .def { color: #66ccff; }
  .item-dot { width: 13px; height: 13px; border-radius: 50%; align-self: center;
    border: 1px solid rgba(0, 0, 0, 0.6); box-shadow: 0 0 5px rgba(0, 0, 0, 0.7); }
  .abil { position: absolute; top: 3px; right: 3px; display: flex; flex-wrap: wrap;
    gap: 2px; max-width: 60%; justify-content: flex-end; }
  .chip { display: inline-block; min-width: 20px; text-align: center;
    font-size: 13px; line-height: 18px; border-radius: 4px; padding: 0 2px;
    background: rgba(8, 8, 12, 0.8); border: 1.5px solid #888; }

  /* hover detail tooltip */
  .tooltip { position: absolute; left: calc(100% + 8px); top: 0; z-index: 30;
    width: 220px; padding: 8px 10px; border-radius: 8px;
    background: #0d0f16; border: 1px solid #3a3f55;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6); color: #ddd;
    font-size: 12px; line-height: 1.4; text-align: left;
    opacity: 0; visibility: hidden; transform: translateY(4px);
    transition: opacity 0.12s ease, transform 0.12s ease; pointer-events: none; }
  .cardwrap:hover .tooltip { opacity: 1; visibility: visible; transform: translateY(0); }
  .tt-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .tt-name { font-weight: 700; font-size: 13px; color: #fff; }
  .tt-cost { color: #6bb8ff; font-weight: 700; white-space: nowrap; }
  .tt-type { color: #99a; text-transform: capitalize; font-size: 11px; margin-top: 1px; }
  .tt-stats { display: flex; gap: 14px; margin: 5px 0; font-weight: 700; }
  .tt-keys { display: flex; flex-direction: column; gap: 3px; margin: 5px 0;
    border-top: 1px solid #2a2f42; padding-top: 5px; }
  .tt-key { display: flex; align-items: center; gap: 6px; }
  .tt-desc { margin-top: 5px; color: #bcbcc8; border-top: 1px solid #2a2f42; padding-top: 5px; }
</style>
