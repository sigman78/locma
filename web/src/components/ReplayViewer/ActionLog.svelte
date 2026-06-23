<!-- web/src/components/ReplayViewer/ActionLog.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import { cardName } from '../../lib/cards'
  import type { Frame } from '../../lib/replay'

  export let frames: Frame[] = []
  export let cursor = 0
  const dispatch = createEventDispatcher<{ seek: number }>()

  function describe(f: Frame): string {
    if (!f.action) return 'opening'
    const s = `P${f.seat}`
    const a = f.action
    if (a.t === 'summon') return `${s} summons ${cardName(a.id)}`
    if (a.t === 'attack') return `${s} attacks ${a.target === -1 ? 'face' : '#' + a.target}`
    if (a.t === 'use') return `${s} uses item`
    return `${s} passes`
  }
</script>

<ul class="log">
  {#each frames as f}
    <li class:active={f.index === cursor} on:click={() => dispatch('seek', f.index)}>
      <span class="turn">T{f.turn ?? '-'}</span> {describe(f)}
    </li>
  {/each}
</ul>

<style>
  .log { list-style: none; margin: 0; padding: 0; max-height: 70vh; overflow: auto;
    font-size: 12px; }
  li { padding: 2px 6px; cursor: pointer; border-radius: 3px; }
  li:hover { background: #1c1c22; }
  li.active { background: #2a2a44; }
  .turn { color: #777; margin-right: 6px; }
</style>
