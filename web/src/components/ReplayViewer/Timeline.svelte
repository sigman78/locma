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
  <button title="previous turn (,)" on:click={() => dispatch('turn', -1)}>⏮</button>
  <button title="step back (←)" on:click={() => dispatch('step', -1)}>◀</button>
  <button class="play" class:playing title="play / pause (space)" on:click={() => dispatch('toggle')}>
    {playing ? '⏸ Pause' : '▶ Play'}
  </button>
  <button title="step forward (→)" on:click={() => dispatch('step', 1)}>▶</button>
  <button title="next turn (.)" on:click={() => dispatch('turn', 1)}>⏭</button>
  <input type="range" min="0" max={length - 1} value={cursor}
    on:input={(e) => dispatch('seek', +(e.currentTarget as HTMLInputElement).value)} />
  <span class="pos">{cursor} / {length - 1}</span>
</div>

<style>
  .timeline { display: flex; gap: 6px; align-items: center; padding: 8px; }
  input[type=range] { flex: 1; accent-color: #4fd97a; }
  .pos { font-size: 13px; color: #aaa; min-width: 64px; text-align: right;
    font-variant-numeric: tabular-nums; }
  button { background: #23232b; color: #ddd; border: 1px solid #333; border-radius: 4px;
    padding: 4px 10px; cursor: pointer; font-size: 14px; }
  button:hover { background: #2c2c38; }
  button.play { background: #2f9e54; color: #06140b; font-weight: 800;
    border-color: #43c873; padding: 5px 18px; font-size: 15px; min-width: 96px; }
  button.play:hover { background: #36b863; }
  button.play.playing { background: #b8863a; border-color: #d8a24a; }
</style>
