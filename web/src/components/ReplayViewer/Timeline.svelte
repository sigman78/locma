<!-- web/src/components/ReplayViewer/Timeline.svelte -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte'
  export let cursor = 0
  export let length = 1
  export let playing = false
  const dispatch = createEventDispatcher<{
    seek: number; toggle: void; step: number; turn: number
  }>()

  function onKey(e: KeyboardEvent) {
    if (e.key === 'ArrowRight') dispatch('step', 1)
    else if (e.key === 'ArrowLeft') dispatch('step', -1)
    else if (e.key === ' ') { e.preventDefault(); dispatch('toggle') }
    else if (e.key === '.') dispatch('turn', 1)
    else if (e.key === ',') dispatch('turn', -1)
  }
  onMount(() => {
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })
</script>

<div class="timeline">
  <button on:click={() => dispatch('turn', -1)}>⏮</button>
  <button on:click={() => dispatch('step', -1)}>◀</button>
  <button on:click={() => dispatch('toggle')}>{playing ? '⏸' : '▶'}</button>
  <button on:click={() => dispatch('step', 1)}>▶</button>
  <button on:click={() => dispatch('turn', 1)}>⏭</button>
  <input type="range" min="0" max={length - 1} value={cursor}
    on:input={(e) => dispatch('seek', +(e.currentTarget as HTMLInputElement).value)} />
  <span class="pos">{cursor} / {length - 1}</span>
</div>

<style>
  .timeline { display: flex; gap: 6px; align-items: center; padding: 8px; }
  input[type=range] { flex: 1; }
  .pos { font-size: 12px; color: #999; min-width: 64px; text-align: right; }
  button { background: #23232b; color: #ddd; border: 1px solid #333; border-radius: 4px;
    padding: 2px 8px; cursor: pointer; }
</style>
