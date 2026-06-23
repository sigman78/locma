<!-- web/src/components/ReplayViewer/Player.svelte -->
<script lang="ts">
  import type { PlayerState } from '../../lib/replay'
  export let player: PlayerState
  export let name: string
  export let seat: 0 | 1

  // deterministic hue from name+seat
  $: hue = ([...`${name}${seat}`].reduce((a, c) => a + c.charCodeAt(0), 0) * 47) % 360
</script>

<div class="player">
  <div class="avatar" style={`background: hsl(${hue} 50% 35%)`}>{name[0]?.toUpperCase() ?? '?'}</div>
  <div class="info">
    <div class="name">{name} <span class="seat">P{seat}</span></div>
    <div class="row">
      <span class="hp">♥ {player.health}</span>
      <span class="mana">◆ {player.mana}/{player.max_mana}</span>
      <span class="rune">⛓ {player.next_rune}</span>
      <span class="deck">🂠 {player.deck_count}</span>
      <span class="hand">✋ {player.hand.length}</span>
    </div>
  </div>
</div>

<style>
  .player { display: flex; gap: 8px; align-items: center; }
  .avatar { width: 40px; height: 40px; border-radius: 50%; display: grid;
    place-items: center; font-weight: 700; color: #fff; }
  .name { font-weight: 600; } .seat { color: #888; font-size: 11px; }
  .row { display: flex; gap: 10px; font-size: 13px; }
  .hp { color: #ff6b6b; } .mana { color: #6bb8ff; } .rune { color: #c9a0ff; }
</style>
