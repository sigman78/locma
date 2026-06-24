<!-- web/src/components/ReplayViewer/Player.svelte -->
<script lang="ts">
  import type { PlayerState } from '../../lib/replay'
  import type { Fx } from '../../lib/fx'
  import { restartAnim } from '../../lib/motion'

  export let player: PlayerState
  export let name: string
  export let seat: 0 | 1
  export let fx: Fx | null = null
  export let fxToken = 0

  // deterministic hue from name+seat
  $: hue = ([...`${name}${seat}`].reduce((a, c) => a + c.charCodeAt(0), 0) * 47) % 360
  $: casting = fx?.cast?.seat === seat
  $: acting = fx?.lunge?.seat === seat || fx?.cast?.seat === seat
  $: faceDmg = fx?.splashes.find((s) => s.target === 'face' && s.seat === seat)?.amount ?? null
</script>

<div class="player" class:acting>
  <div
    class="avatar"
    class:acting
    style={`background: hsl(${hue} 50% 35%)`}
    use:restartAnim={{ cls: casting ? 'cast-pulse' : null, token: fxToken }}
  >
    {name[0]?.toUpperCase() ?? '?'}
  </div>
  <div class="info">
    <div class="name">{name} <span class="seat">P{seat}</span></div>
    <div class="row">
      <span class="hp">
        ♥ {player.health}
        {#key fxToken}
          {#if faceDmg != null}<span class="locma-dmg face">-{faceDmg}</span>{/if}
        {/key}
      </span>
      <span class="mana">◆ {player.mana}/{player.max_mana}</span>
      <span class="rune">⛓ {player.next_rune}</span>
      <span class="deck">🂠 {player.deck_count}</span>
      <span class="hand">✋ {player.hand.length}</span>
    </div>
  </div>
</div>

<style>
  .player { display: flex; gap: 12px; align-items: center; padding: 4px 8px;
    border-radius: 8px; border: 1px solid transparent; transition: border-color 0.2s; }
  .player.acting { border-color: #ffd23d66; background: rgba(255, 210, 61, 0.06); }
  .avatar { width: 56px; height: 56px; border-radius: 50%; display: grid;
    place-items: center; font-weight: 700; color: #fff; font-size: 26px;
    transition: box-shadow 0.15s; }
  .avatar.acting { box-shadow: 0 0 0 3px #ffd23d, 0 0 14px 2px rgba(255, 210, 61, 0.8); }
  .name { font-weight: 600; font-size: 20px; } .seat { color: #888; font-size: 15px; }
  .row { display: flex; gap: 16px; font-size: 19px; align-items: center; }
  .hp { color: #ff6b6b; position: relative; } .mana { color: #6bb8ff; }
  .rune { color: #c9a0ff; }
  .locma-dmg.face { font-size: 30px; top: -12px; }
</style>
