<!-- web/src/components/ReplayViewer/Player.svelte -->
<script lang="ts">
  import type { PlayerState } from '../../lib/replay'
  import type { Fx } from '../../lib/fx'
  import { restartAnim } from '../../lib/motion'

  export let player: PlayerState
  export let name: string
  export let seat: 0 | 1
  export let active = false
  export let fx: Fx | null = null
  export let fxToken = 0

  // deterministic hue from name+seat
  $: hue = ([...`${name}${seat}`].reduce((a, c) => a + c.charCodeAt(0), 0) * 47) % 360
  $: casting = fx?.cast?.seat === seat
  $: acting = fx?.lunge?.seat === seat || fx?.cast?.seat === seat
  $: faceDmg = fx?.splashes.find((s) => s.target === 'face' && s.seat === seat)?.amount ?? null
</script>

<div class="player">
  <div
    class="avatar"
    class:acting
    style={`background: hsl(${hue} 50% 35%)`}
    use:restartAnim={{ cls: casting ? 'cast-pulse' : null, token: fxToken }}
  >
    {name[0]?.toUpperCase() ?? '?'}
  </div>
  <div class="info">
    <div class="name">
      {name} <span class="seat">P{seat}</span>
      {#if active}<span class="turn-badge">● TURN</span>{/if}
    </div>
    <div class="row">
      <span class="hp">
        ♥ {player.health}
        {#key fxToken}
          {#if faceDmg != null}<span class="locma-dmg face">-{faceDmg}</span>{/if}
        {/key}
      </span>
      <span class="mana">◆ {player.mana}/{player.max_mana}</span>
      {#if player.bonus_draw > 0}
        <span class="draw" title="extra cards drawn next turn (damage taken)">
          +{player.bonus_draw}🂠
        </span>
      {/if}
      <span class="deck">🂠 {player.deck_count}</span>
      <span class="hand">✋ {player.hand.length}</span>
    </div>
  </div>
</div>

<style>
  .player { display: flex; gap: 12px; align-items: center; padding: 6px 10px;
    text-align: left; }
  .info { text-align: left; }
  .avatar { width: 56px; height: 56px; border-radius: 50%; display: grid;
    place-items: center; font-weight: 700; color: #fff; font-size: 26px;
    transition: box-shadow 0.15s; }
  /* transient action pop (the actual move being played) */
  .avatar.acting { box-shadow: 0 0 0 3px #ffd23d, 0 0 16px 3px rgba(255, 210, 61, 0.9); }
  .name { font-weight: 600; font-size: 20px; text-align: left; }
  .seat { color: #888; font-size: 15px; }
  .turn-badge { color: #ffd23d; font-size: 12px; font-weight: 800; letter-spacing: 0.5px;
    background: rgba(255, 210, 61, 0.15); border: 1px solid #ffd23d66;
    border-radius: 10px; padding: 1px 7px; margin-left: 6px; vertical-align: middle; }
  .row { display: flex; gap: 16px; font-size: 19px; align-items: center; }
  .hp { color: #ff6b6b; position: relative; } .mana { color: #6bb8ff; }
  .draw { color: #7ddf7d; font-weight: 700; }
  .locma-dmg.face { font-size: 30px; top: -12px; }
</style>
