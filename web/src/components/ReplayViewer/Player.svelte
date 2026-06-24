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
  $: faceDmg = fx?.splashes.find((s) => s.target === 'face' && s.seat === seat)?.amount ?? null
</script>

<div class="player">
  <div
    class="avatar"
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
  .player { display: flex; gap: 12px; align-items: center; }
  .avatar { width: 56px; height: 56px; border-radius: 50%; display: grid;
    place-items: center; font-weight: 700; color: #fff; font-size: 26px; }
  .name { font-weight: 600; font-size: 20px; } .seat { color: #888; font-size: 15px; }
  .row { display: flex; gap: 16px; font-size: 19px; align-items: center; }
  .hp { color: #ff6b6b; position: relative; } .mana { color: #6bb8ff; }
  .rune { color: #c9a0ff; }
  .locma-dmg.face { font-size: 18px; top: -4px; }
</style>
