<!-- web/src/components/ReplayViewer/DraftPanel.svelte -->
<script lang="ts">
  import { cardName } from '../../lib/cards'
  import type { Replay } from '../../lib/replay'

  export let draft: Replay['draft']
  // pick index per (round, seat)
  function pickIdx(round: number, seat: number): number | null {
    const p = draft.picks.find((x) => x.round === round && x.seat === seat)
    return p ? p.pick : null
  }
</script>

<div class="draft">
  {#each draft.pool as triplet, round}
    <div class="round">
      <span class="rno">{round + 1}</span>
      {#each triplet as cid, i}
        <span class="opt"
          class:p0={pickIdx(round, 0) === i}
          class:p1={pickIdx(round, 1) === i}>{cardName(cid)}</span>
      {/each}
    </div>
  {/each}
</div>

<style>
  .draft { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px;
    max-height: 70vh; overflow: auto; padding: 8px; }
  .round { display: flex; gap: 6px; align-items: center; font-size: 12px; }
  .rno { color: #777; width: 22px; }
  .opt { padding: 1px 5px; border-radius: 3px; background: #1c1c22; border: 1px solid #2a2a33; }
  .opt.p0 { border-color: #ffcc55; }
  .opt.p1 { border-color: #6bb8ff; }
  .opt.p0.p1 { border-image: linear-gradient(90deg,#ffcc55,#6bb8ff) 1; }
</style>
