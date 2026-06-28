<script lang="ts">
  import type { CardState } from '../../lib/replay'
  import { artUrl, cardName, card as cardMeta } from '../../lib/cards'
  import { abilityList, auraSplit } from '../../lib/abilities'
  import { restartAnim } from '../../lib/motion'
  import Tooltip from '../ReplayViewer/Tooltip.svelte'

  export let card: CardState
  export let facing: 'up' | 'down' | null = null
  export let slideX = 0
  export let slideY = 0
  export let flash = false
  export let hit = false
  export let dying = false
  export let dmgDelay = false
  export let damage: number | null = null
  export let dim = false
  export let fxToken = 0

  let imgOk = true
  $: name = cardName(card.card_id)
  $: meta = cardMeta(card.card_id)
  $: baseAbil = abilityList(meta?.abilities)
  $: baseLetters = new Set([...(meta?.abilities ?? '')].filter((ch) => ch !== '-'))
  $: atkDelta = meta ? card.atk - meta.attack : 0
  $: defDelta = meta ? card.def - meta.defense : 0
  $: split = auraSplit(card.abilities)
  $: sliding = slideX !== 0 || slideY !== 0
  $: animCls = sliding ? 'sliding' : flash ? 'flashing' : null
  $: slideStyle = sliding ? `--sx:${slideX}px; --sy:${slideY}px;` : ''
  $: tip = facing === 'down' ? ('below' as const) : ('above' as const)
  $: baseAtk = meta ? meta.attack : card.atk
  $: baseDef = meta ? meta.defense : card.def

  const ITEM = {
    itemgreen: { color: '#4fd97a', emoji: '🟢', label: 'Green item' },
    itemred: { color: '#ff5d5d', emoji: '🔴', label: 'Red item' },
    itemblue: { color: '#5aa9ff', emoji: '🔵', label: 'Blue item' },
  } as const
  $: item = meta ? (ITEM as Record<string, { color: string; emoji: string; label: string }>)[meta.type] : undefined
  $: typeLabel = meta ? (item ? `${item.emoji} ${item.label}` : 'Creature') : ''
</script>

<!-- root carries the aura class hooks (visuals are Task M2) -->
<div class="minionwrap" class:taunt={split.taunt} class:ward={split.ward} class:lethal={split.lethal}>
  <div
    class="minion"
    class:face-up={facing === 'up'}
    class:face-down={facing === 'down'}
    class:attacking={sliding}
    class:attacked={card.has_attacked}
    class:dim
    style={slideStyle}
    use:restartAnim={{ cls: animCls, token: fxToken }}
  >
    {#if imgOk}
      <img src={artUrl(card.card_id)} alt={name} draggable="false" on:error={() => (imgOk = false)} />
    {:else}
      <div class="placeholder"><span class="nm">{name}</span></div>
    {/if}

    <!-- B/C/D keyword pills — aura keywords (G/L/W) are class-only hooks for M2 -->
    <div class="abil">
      {#each split.pills as a}
        <span
          class="chip"
          class:granted={!baseLetters.has(a.letter)}
          style={`border-color:${a.color}`}
          title={baseLetters.has(a.letter) ? a.name : `${a.name} (granted)`}>{a.emoji}</span>
      {/each}
    </div>

    <!-- stat mini-plates: atk bottom-left, def bottom-right -->
    <div class="atk-plate" class:buffed={atkDelta > 0} class:reduced={atkDelta < 0}>{card.atk}</div>
    <div class="def-plate" class:buffed={defDelta > 0} class:reduced={defDelta < 0}>{card.def}</div>

    <!-- transient combat overlays (reuse global CSS from app.css) -->
    {#key fxToken}
      {#if flash}<div class="flash-blob"></div>{/if}
      {#if hit}<div class="hit-flash" class:delayed={dmgDelay}></div>{/if}
      {#if damage != null}<div class="locma-dmg" class:delayed={dmgDelay}>-{damage}</div>{/if}
    {/key}
    {#if dying}<div class="death-cross">✕</div>{/if}
  </div>

  <!-- sleeping overlay sits above the dimmed minion -->
  {#if dim && !card.has_attacked}<div class="sleep" title="summoning sick — can't attack yet">💤</div>{/if}

  <Tooltip {name} {meta} {baseAbil} {typeLabel} {tip} {baseAtk} {baseDef} />
</div>

<style>
  .minionwrap { position: relative; width: var(--card-w, 108px); height: var(--card-h, 150px);
    user-select: none; -webkit-user-select: none; -webkit-user-drag: none; }
  .minionwrap:hover { z-index: 50; }

  /* frameless sprite — no card background, border, or crosshatch */
  .minion { position: relative; width: 100%; height: 100%;
    user-select: none; -webkit-user-select: none; -webkit-user-drag: none; }
  .minion img { width: 100%; height: 100%; object-fit: cover; border-radius: 4px; }
  .placeholder { display: grid; place-items: center; height: 100%; padding: 4px;
    text-align: center; font-size: 13px; color: #ddd; }

  /* stat mini-plates: small dark pill at bottom corners */
  .atk-plate, .def-plate { position: absolute; bottom: 4px; z-index: 2;
    background: rgba(0,0,0,0.7); border-radius: 5px; padding: 1px 6px;
    font-weight: 700; font-size: 15px; pointer-events: none; }
  .atk-plate { left: 4px; color: #ffcc55; }
  .def-plate { right: 4px; color: #66ccff; }
  .buffed { color: #4fd97a; text-shadow: 0 0 6px rgba(79, 217, 122, 0.7); }
  .reduced { color: #ff6b6b; text-shadow: 0 0 6px rgba(255, 107, 107, 0.7); }

  /* B/C/D keyword pills — top-right cluster */
  .abil { position: absolute; top: 3px; right: 3px; display: flex; flex-wrap: wrap;
    gap: 2px; max-width: 60%; justify-content: flex-end; }
  .chip { display: inline-block; min-width: 20px; text-align: center;
    font-size: 13px; line-height: 18px; border-radius: 4px; padding: 0 2px;
    background: rgba(8, 8, 12, 0.8); border: 1.5px solid #888; }
  .chip.granted { border-style: dashed; background: rgba(79, 217, 122, 0.18);
    box-shadow: 0 0 7px rgba(79, 217, 122, 0.8); }

  /* sleeping (summoning-sick) indicator */
  .sleep { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 40px; z-index: 3; pointer-events: none; filter: drop-shadow(0 2px 3px #000); }

  /* combat state filters */
  .minion.attacked { filter: saturate(0.6); }
  .minion.dim { opacity: 0.5; }
  /* attacker highlight — declared after .attacked so it wins */
  .minion.attacking { outline: 3px solid #ffd23d; outline-offset: 2px;
    filter: brightness(1.18); opacity: 1; z-index: 4; border-radius: 4px; }

  /* reveal the shared Tooltip on hover */
  .minionwrap:hover :global(.tooltip) { opacity: 1; visibility: visible; transform: translateX(-50%) translateY(0); }
</style>
