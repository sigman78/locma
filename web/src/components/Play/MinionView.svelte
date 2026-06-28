<script lang="ts">
  import type { CardState } from '../../lib/replay'
  import { artUrl, cardName, card as cardMeta, creatureSpecial } from '../../lib/cards'
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
  // a generic on-summon effect → ✨ pill on the face (detail in the tooltip)
  $: special = meta ? creatureSpecial(meta.description) : ''
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

<!-- root carries the aura class hooks; M2 implements the three aura visuals below -->
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
    <!--
      sprite-stack: local isolated stacking context for aura layering.
      isolation: isolate lets z-index: -1 (taunt shield) sit behind the sprite
      and z-index: 1 (ward bubble) sit in front, without disturbing the outer
      z-index chain (.minion.attacking z-index:4 still pops above neighbor slots).

      .minion also has isolation: isolate (Fix 2): this contains all child z-indices
      (stat plates z:2, overlays z:3–7) within .minion's own stacking context, preventing
      them from escaping to the root SC where they would paint over hand-card tooltips.
      .minion.attacking { z-index:4 } still works — that sets the SC's own z-index
      within .minionwrap, letting the attacker pop above neighbour slots as before.

      z-order inside sprite-stack (back → front):
        taunt shield  (z:-1)  → sprite img  (block, auto)  → ward bubble  (z:1)

      z-order in .minion (back → front):
        sprite-stack (z:auto) → stat plates/pills (z:2) → sleep (z:3)
        → hit-flash (z:4) → flash-blob (z:5) → locma-dmg (z:6) → death-cross (z:7)
    -->
    <div class="sprite-stack">
      <!-- M2: Taunt (G) — heater-shield SVG behind the sprite, full card-slot size -->
      {#if split.taunt}
      <svg
        class="taunt-shield"
        viewBox="0 0 100 130"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id={`taunt-steel-${card.iid}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#a6adb6" />
            <stop offset="100%" stop-color="#565d67" />
          </linearGradient>
        </defs>
        <path
          d="M8,5 L92,5 L92,78 Q92,115 50,126 Q8,115 8,78 Z"
          fill={`url(#taunt-steel-${card.iid})`}
          fill-opacity="0.9"
          stroke="#33383f"
          stroke-width="3"
          stroke-linejoin="round"
        />
      </svg>
      {/if}

      {#if imgOk}
        <!-- M2: Lethal (L) glow applied via .lethal img CSS (drop-shadow traces alpha cutout) -->
        <img src={artUrl(card.card_id)} alt={name} draggable="false" on:error={() => (imgOk = false)} />
      {:else}
        <div class="placeholder"><span class="nm">{name}</span></div>
      {/if}

      <!-- M2: Ward (W) — light-blue pulsing barrier bubble over the sprite, under stat plates -->
      {#if split.ward}
      <div class="ward-bubble" aria-hidden="true"></div>
      {/if}
    </div>

    <!-- B/C/D keyword pills (G/L/W are aura visuals) + ✨ special-effect pill -->
    {#if split.pills.length || special}
      <div class="abil">
        {#if special}<span class="chip special" title="special effect — hover for details">✨</span>{/if}
        {#each split.pills as a}
          <span
            class="chip"
            class:granted={!baseLetters.has(a.letter)}
            style={`border-color:${a.color}`}
            title={baseLetters.has(a.letter) ? a.name : `${a.name} (granted)`}>{a.emoji}</span>
        {/each}
      </div>
    {/if}

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

  /* frameless sprite — no card background, border, or crosshatch.
     isolation: isolate makes .minion a stacking context so all child z-indices
     (stat plates z:2, overlays z:3–7) are contained here and do not escape to the
     root SC where they would paint over hand-card tooltips (Fix 2). */
  .minion { position: relative; width: 100%; height: 100%;
    user-select: none; -webkit-user-select: none; -webkit-user-drag: none;
    isolation: isolate; }

  /* sprite-stack: isolated stacking context so taunt (z:-1) and ward (z:1) stay
     contained without touching the outer .minionwrap / .minion z-index chain */
  .sprite-stack { position: relative; width: 100%; height: 100%; isolation: isolate; }
  .sprite-stack img { width: 100%; height: 100%; object-fit: cover; border-radius: 4px; }
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
  /* special on-summon effect indicator — amber attention pill */
  .chip.special { border-color: #ffd23d; background: rgba(255, 210, 61, 0.2);
    box-shadow: 0 0 7px rgba(255, 210, 61, 0.6); }

  /* sleeping (summoning-sick) indicator */
  .sleep { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 40px; z-index: 3; pointer-events: none; filter: drop-shadow(0 2px 3px #000); }

  /* combat state filters — applied to .minion, so they compose with img's lethal glow */
  .minion.attacked { filter: saturate(0.6); }
  .minion.dim { opacity: 0.5; }
  /* attacker highlight — declared after .attacked so it wins */
  .minion.attacking { outline: 3px solid #ffd23d; outline-offset: 2px;
    filter: brightness(1.18); opacity: 1; z-index: 4; border-radius: 4px; }

  /* reveal the shared Tooltip on hover */
  .minionwrap:hover :global(.tooltip) { opacity: 1; visibility: visible; transform: translateX(-50%) translateY(0); }

  /* ── M2: Aura keyword visuals ─────────────────────────────────────────────────── */

  /* Taunt (G): heater-shield SVG behind the sprite.
     z-index: -1 is contained within .sprite-stack (isolation: isolate), so it stays
     behind the sprite img without bleeding below the .minionwrap background. */
  .taunt-shield {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    z-index: -1;
    pointer-events: none;
  }

  /* Lethal (L): static green silhouette glow on the sprite img.
     drop-shadow() traces the alpha cutout of the creature.
     Existing .minion filters (saturate / brightness) apply to the whole .minion
     on top of this — lethal glow naturally dims when attacked / brightens when attacking. */
  .lethal .sprite-stack img {
    filter: drop-shadow(0 0 6px #4fd97a) drop-shadow(0 0 2px #4fd97a);
  }

  /* Ward (W): light-blue pulsing barrier bubble over the sprite, under stat plates.
     mix-blend-mode: screen adds light rather than graying the sprite.
     isolation: isolate on .sprite-stack confines the blend to sprite-stack content,
     so the bubble does NOT lighten stats or pills (which live outside .sprite-stack).
     border-radius: 50% gives the elliptical bubble shape (Fix 1). */
  .ward-bubble {
    position: absolute; inset: 0; z-index: 1;
    border-radius: 50%; pointer-events: none;
    box-shadow: inset 0 0 14px 3px rgba(150, 235, 255, 0.85);
    background: radial-gradient(
      circle at 50% 45%,
      rgba(150, 235, 255, 0.18) 0%,
      rgba(100, 200, 255, 0.08) 55%,
      transparent 80%
    );
    mix-blend-mode: screen;
    animation: ward-pulse 2s ease-in-out infinite;
  }

  @keyframes ward-pulse {
    0%, 100% { opacity: 0.75; transform: scale(1);    }
    50%       { opacity: 1;    transform: scale(1.02); }
  }
</style>
