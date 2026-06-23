<!-- web/src/components/ReplayViewer/ReplayViewer.svelte -->
<script lang="ts">
  import { onDestroy } from 'svelte'
  import { Playback, type Replay } from '../../lib/replay'
  import ActionLog from './ActionLog.svelte'
  import Board from './Board.svelte'
  import DraftPanel from './DraftPanel.svelte'
  import Timeline from './Timeline.svelte'

  export let replay: Replay

  let tab: 'draft' | 'battle' = 'battle'
  let pb = new Playback(replay)
  let cursor = 0
  let playing = false
  let timer: ReturnType<typeof setInterval> | null = null

  $: nameA = replay.header.a_seat === 0 ? replay.header.policy_a : replay.header.policy_b
  $: nameB = replay.header.a_seat === 0 ? replay.header.policy_b : replay.header.policy_a

  function sync() { cursor = pb.cursor }
  function seek(i: number) { pb.seek(i); sync() }
  function step(d: number) { d > 0 ? pb.next() : pb.prev(); sync() }
  function turn(d: number) { d > 0 ? pb.nextTurn() : pb.prevTurn(); sync() }
  function toggle() {
    playing = !playing
    if (playing) {
      timer = setInterval(() => {
        if (pb.cursor >= pb.frames.length - 1) { toggle(); return }
        pb.next(); sync()
      }, 600)
    } else if (timer) { clearInterval(timer); timer = null }
  }
  onDestroy(() => { if (timer) clearInterval(timer) })
</script>

<div class="viewer">
  <header>
    <strong>{replay.header.policy_a}</strong> vs <strong>{replay.header.policy_b}</strong>
    · seed {replay.header.seed} · winner P{replay.header.winner} · {replay.header.turns} turns
    <span class="tabs">
      <button class:on={tab === 'draft'} on:click={() => (tab = 'draft')}>Draft</button>
      <button class:on={tab === 'battle'} on:click={() => (tab = 'battle')}>Battle</button>
    </span>
  </header>

  {#if tab === 'draft'}
    <DraftPanel draft={replay.draft} />
  {:else}
    <div class="battle">
      <div class="main">
        <Board snapshot={pb.current.snapshot} {nameA} {nameB} />
        <Timeline {cursor} length={pb.frames.length} {playing}
          on:seek={(e) => seek(e.detail)} on:step={(e) => step(e.detail)}
          on:turn={(e) => turn(e.detail)} on:toggle={toggle} />
      </div>
      <aside><ActionLog frames={pb.frames} {cursor} on:seek={(e) => seek(e.detail)} /></aside>
    </div>
  {/if}
</div>

<style>
  .viewer { color: #ddd; }
  header { padding: 8px; font-size: 14px; }
  .tabs { margin-left: 12px; }
  .tabs button { background: #23232b; color: #ddd; border: 1px solid #333;
    padding: 2px 10px; cursor: pointer; }
  .tabs button.on { background: #2a2a44; }
  .battle { display: flex; gap: 12px; }
  .main { flex: 1; } aside { width: 260px; }
</style>
