<!-- web/src/components/ReplayViewer/ActionLog.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import { cardName } from '../../lib/cards'
  import type { Frame } from '../../lib/replay'

  export let frames: Frame[] = []
  export let cursor = 0
  export let cardIds = new Map<number, number>()
  const dispatch = createEventDispatcher<{ seek: number }>()

  function nm(iid: number): string {
    const cid = cardIds.get(iid)
    return cid != null ? cardName(cid) : `#${iid}`
  }

  function describe(f: Frame): string {
    if (!f.action) return 'opening'
    const s = `P${f.seat}`
    const a = f.action
    if (a.t === 'summon') return `${s} summons ${nm(a.id)}`
    if (a.t === 'attack') return `${s} attacks ${a.target === -1 ? 'face' : '#' + a.target}`
    if (a.t === 'use') return `${s} uses ${nm(a.item)}`
    return `${s} passes`
  }
</script>

<ul class="log">
  {#each frames as f}
    <li class:active={f.index === cursor} on:click={() => dispatch('seek', f.index)}>
      <span class="turn">T{f.turn ?? '-'}</span><span class="desc">{describe(f)}</span>
    </li>
  {/each}
</ul>

<style>
  .log { list-style: none; margin: 0; padding: 0; max-height: 78vh; overflow: auto;
    font-size: 11px; text-align: left; line-height: 1.35; }
  li { display: flex; gap: 6px; align-items: baseline; padding: 1px 6px; cursor: pointer;
    border-radius: 3px; }
  li:hover { background: #1c1c22; }
  li.active { background: #2a2a44; }
  .turn { color: #777; flex: 0 0 24px; }
  .desc { flex: 1 1 auto; }
</style>
