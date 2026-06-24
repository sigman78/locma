<!-- web/src/components/ReplayViewer/Board.svelte -->
<script lang="ts">
  import type { Snapshot } from '../../lib/replay'
  import CardView from './CardView.svelte'
  import Hand from './Hand.svelte'
  import Player from './Player.svelte'

  export let snapshot: Snapshot
  export let nameA: string  // seat 0
  export let nameB: string  // seat 1
  $: p0 = snapshot.players[0]
  $: p1 = snapshot.players[1]
</script>

<div class="board">
  <Player player={p1} name={nameB} seat={1} />
  <Hand cards={p1.hand} faceUp={true} />
  <div class="field top">
    {#each p1.board as c (c.iid)}
      <div class:sick={c.can_attack === false}><CardView card={c} /></div>
    {/each}
  </div>
  <hr />
  <div class="field bottom">
    {#each p0.board as c (c.iid)}
      <div class:sick={c.can_attack === false}><CardView card={c} /></div>
    {/each}
  </div>
  <Hand cards={p0.hand} faceUp={true} />
  <Player player={p0} name={nameA} seat={0} />
</div>

<style>
  .board { display: flex; flex-direction: column; gap: 8px; padding: 12px;
    background: #15151b; border-radius: 8px; }
  .field { display: flex; gap: 6px; min-height: 104px; align-items: center;
    padding: 4px; background: #101015; border-radius: 6px; }
  .sick { opacity: 0.55; }
  hr { width: 100%; border: none; border-top: 1px dashed #333; margin: 2px 0; }
</style>
