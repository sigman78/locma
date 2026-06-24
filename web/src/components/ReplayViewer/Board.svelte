<!-- web/src/components/ReplayViewer/Board.svelte -->
<script lang="ts">
  import type { Snapshot } from '../../lib/replay'
  import type { Fx } from '../../lib/fx'
  import { deathFx } from '../../lib/motion'
  import CardView from './CardView.svelte'
  import Hand from './Hand.svelte'
  import Player from './Player.svelte'

  export let snapshot: Snapshot
  export let nameA: string // seat 0
  export let nameB: string // seat 1
  export let fx: Fx | null = null
  export let fxToken = 0
  $: p0 = snapshot.players[0]
  $: p1 = snapshot.players[1]

  function dmg(seat: number, iid: number): number | null {
    const s = fx?.splashes.find((x) => x.seat === seat && x.target === iid && !x.fatal)
    return s ? s.amount : null
  }
  function lungeDir(seat: number, iid: number): 'up' | 'down' | null {
    if (fx?.lunge && fx.lunge.seat === seat && fx.lunge.iid === iid) {
      return seat === 0 ? 'up' : 'down'
    }
    return null
  }
</script>

<div class="board">
  <Player player={p1} name={nameB} seat={1} {fx} {fxToken} />
  <Hand cards={p1.hand} faceUp={true} />
  <div class="field top">
    {#each p1.board as c (c.iid)}
      <div class:sick={c.can_attack === false} out:deathFx>
        <CardView card={c} lunge={lungeDir(1, c.iid)} damage={dmg(1, c.iid)} {fxToken} />
      </div>
    {/each}
  </div>
  <hr />
  <div class="field bottom">
    {#each p0.board as c (c.iid)}
      <div class:sick={c.can_attack === false} out:deathFx>
        <CardView card={c} lunge={lungeDir(0, c.iid)} damage={dmg(0, c.iid)} {fxToken} />
      </div>
    {/each}
  </div>
  <Hand cards={p0.hand} faceUp={true} />
  <Player player={p0} name={nameA} seat={0} {fx} {fxToken} />
</div>

<style>
  .board { display: inline-flex; flex-direction: column; gap: 10px; padding: 14px;
    background: #15151b; border-radius: 8px;
    --card-w: 108px; --card-h: 150px; --gap: 8px; --board-cols: 6; --hand-cols: 8; }
  .field { display: flex; gap: var(--gap); align-items: center; justify-content: flex-start;
    padding: 6px; background: #101015; border-radius: 6px;
    width: calc(var(--board-cols) * var(--card-w) + (var(--board-cols) - 1) * var(--gap) + 12px);
    min-height: calc(var(--card-h) + 12px); }
  .field > div { flex: 0 0 auto; }
  .sick { opacity: 0.55; }
  hr { width: 100%; border: none; border-top: 1px dashed #333; margin: 2px 0; }
</style>
