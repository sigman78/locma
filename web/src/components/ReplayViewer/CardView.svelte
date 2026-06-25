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
  export let tipDir: 'above' | 'below' | null = null // hover-tooltip placement override
  export let slideX = 0 // px toward an attack target (Play only); 0 = no slide
  export let slideY = 0
  export let flash = false // cast/use flash over this card
  export let dying = false // red cross then removal
  export let dmgDelay = false // delay the damage number so it lands after the slide

  let imgOk = true
  $: name = cardName(card.card_id)
  $: meta = cardMeta(card.card_id)
  // face shows the live in-play state (incl. buffs); tooltip shows the printed card
  $: abil = abilityList(card.abilities)
  $: baseAbil = abilityList(meta?.abilities)
  // letters present on the printed card — anything extra on the face is granted by an effect
  $: baseLetters = new Set([...(meta?.abilities ?? '')].filter((ch) => ch !== '-'))
  $: atkDelta = meta ? card.atk - meta.attack : 0
  $: defDelta = meta ? card.def - meta.defense : 0
  $: guard = hasAura(card.abilities, 'G')
  $: ward = hasAura(card.abilities, 'W')
  $: lethal = hasAura(card.abilities, 'L')
  // Play uses a measured slide; the ReplayViewer uses up/down lunge. Slide wins when set.
  $: sliding = slideX !== 0 || slideY !== 0
  $: animCls = sliding ? 'sliding' : flash ? 'flashing' : lunge ? `lunge-${lunge}` : null
  $: slideStyle = sliding ? `--sx:${slideX}px; --sy:${slideY}px;` : ''
  // tooltip sits above the card by default, below for opponent (top-row) cards,
  // so it never covers a horizontal neighbour; callers can override via tipDir.
  $: tip = tipDir ?? (facing === 'down' ? 'below' : 'above')

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
      class:attacking={!!lunge || sliding}
      class:attacked={card.has_attacked}
      class:dim
      style={slideStyle}
      use:restartAnim={{ cls: animCls, token: fxToken }}
    >
      {#if imgOk}
        <img src={artUrl(card.card_id)} alt={name} on:error={() => (imgOk = false)} />
      {:else}
        <div class="placeholder"><span class="nm">{name}</span></div>
      {/if}
      {#if showAuras && ward}<div class="ward-tint"></div>{/if}
      {#if meta}<div class="cost" title="mana cost">◆ {meta.cost}</div>{/if}
      <div class="stats">
        <span class="atk" class:buffed={atkDelta > 0} class:reduced={atkDelta < 0}>{card.atk}</span>
        {#if item}<span class="item-dot" style={`background:${item.color}`} title={item.label}></span>{/if}
        <span class="def" class:buffed={defDelta > 0} class:reduced={defDelta < 0}>{card.def}</span>
      </div>
      <div class="abil">
        {#each abil as a}
          <span
            class="chip"
            class:granted={!baseLetters.has(a.letter)}
            style={`border-color:${a.color}`}
            title={baseLetters.has(a.letter) ? a.name : `${a.name} (granted)`}>{a.emoji}</span>
        {/each}
      </div>
      {#key fxToken}
        {#if flash}<div class="flash-blob"></div>{/if}
        {#if damage != null}<div class="locma-dmg" class:delayed={dmgDelay}>-{damage}</div>{/if}
      {/key}
      {#if dying}<div class="death-cross">✕</div>{/if}
    </div>

    {#if dim && !card.has_attacked}<div class="sleep" title="summoning sick — can't attack yet">💤</div>{/if}

    <div class="tooltip" class:tip-above={tip === 'above'} class:tip-below={tip === 'below'}>
      <div class="tt-head">
        <span class="tt-name">{meta?.name ?? name}</span>
        {#if meta}<span class="tt-cost">◆ {meta.cost}</span>{/if}
      </div>
      {#if meta}<div class="tt-type">{typeLabel}</div>{/if}
      <!-- tooltip mirrors the printed card (base stats), not the in-play buffed state -->
      <div class="tt-stats">
        <span class="atk">⚔ {meta ? meta.attack : card.atk}</span>
        <span class="def">🛡 {meta ? meta.defense : card.def}</span>
      </div>
      {#if baseAbil.length}
        <div class="tt-keys">
          {#each baseAbil as a}
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
  .cardwrap:hover { z-index: 50; }
  /* mana cost — top-left gem (printed cost) */
  .cost { position: absolute; top: 3px; left: 3px; z-index: 2;
    min-width: 20px; text-align: center; font-size: 13px; line-height: 18px;
    font-weight: 700; color: #cfe6ff; background: rgba(8, 12, 24, 0.85);
    border: 1.5px solid #5a7fd0; border-radius: 4px; padding: 0 3px; }
  /* sleeping (summoning-sick) indicator — sits above the dimmed card */
  .sleep { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 40px; z-index: 3; pointer-events: none; filter: drop-shadow(0 2px 3px #000); }
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
  /* Ward — bright shining protective bubble (screen-blend so it adds light, not gray) */
  .ward-tint { position: absolute; inset: 0; pointer-events: none; border-radius: 6px;
    mix-blend-mode: screen;
    background: radial-gradient(ellipse at 50% 45%,
      rgba(180, 245, 255, 0.6), rgba(120, 220, 255, 0.2) 55%, transparent 78%);
    box-shadow: inset 0 0 22px 5px rgba(150, 235, 255, 0.95); }
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
  /* live stat deviations from the printed card: green = buffed, red = reduced/damaged */
  .stats .buffed { color: #4fd97a; text-shadow: 0 0 6px rgba(79, 217, 122, 0.7); }
  .stats .reduced { color: #ff6b6b; text-shadow: 0 0 6px rgba(255, 107, 107, 0.7); }
  .item-dot { width: 13px; height: 13px; border-radius: 50%; align-self: center;
    border: 1px solid rgba(0, 0, 0, 0.6); box-shadow: 0 0 5px rgba(0, 0, 0, 0.7); }
  .abil { position: absolute; top: 3px; right: 3px; display: flex; flex-wrap: wrap;
    gap: 2px; max-width: 60%; justify-content: flex-end; }
  .chip { display: inline-block; min-width: 20px; text-align: center;
    font-size: 13px; line-height: 18px; border-radius: 4px; padding: 0 2px;
    background: rgba(8, 8, 12, 0.8); border: 1.5px solid #888; }
  /* abilities granted in-play (not on the printed card) read as a glowing buff */
  .chip.granted { border-style: dashed; background: rgba(79, 217, 122, 0.18);
    box-shadow: 0 0 7px rgba(79, 217, 122, 0.8); }

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
  .cardwrap:hover .tooltip { opacity: 1; visibility: visible; transform: translateX(-50%) translateY(0); }
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
