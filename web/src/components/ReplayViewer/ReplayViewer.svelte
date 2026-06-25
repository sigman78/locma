<!-- web/src/components/ReplayViewer/ReplayViewer.svelte -->
<script lang="ts">
  import { createEventDispatcher, onDestroy } from 'svelte'
  import { Playback, type CardState, type EventDict, type Replay, type Snapshot } from '../../lib/replay'
  import { computeFx, type Fx } from '../../lib/fx'
  import { animate, pulse } from '../../lib/motion'
  import ActionLog from './ActionLog.svelte'
  import Board from './Board.svelte'
  import DraftPanel from './DraftPanel.svelte'
  import Timeline from './Timeline.svelte'

  export let replay: Replay

  const dispatch = createEventDispatcher<{ back: void }>()
  let tab: 'draft' | 'battle' = 'battle'
  let pb = new Playback(replay)
  let cursor = 0
  let playing = false
  let timer: ReturnType<typeof setInterval> | null = null
  let fx: Fx | null = null
  let fxToken = 0
  // dying minions retained for the cross/removal animation: their pre-death CardState
  // (read from the previous frame, where they were still on the board) + slot index.
  let dyingCards: { seat: number; card: CardState; index: number }[] = []

  $: nameA = replay.header.a_seat === 0 ? replay.header.policy_a : replay.header.policy_b
  $: nameB = replay.header.a_seat === 0 ? replay.header.policy_b : replay.header.policy_a
  $: snapshot = pb.frames[cursor]?.snapshot

  function clearFx() { fx = null; dyingCards = []; animate.set(false) }

  /** Cards that died this step, read from `prev` (where they were still on the board)
   *  so the replay can show a death cross before the unit leaves its slot. */
  function dyingFrom(events: EventDict[], prev: Snapshot): typeof dyingCards {
    const out: typeof dyingCards = []
    for (const e of events) {
      if (e.t !== 'unit_died') continue
      const board = prev.players[e.seat]?.board ?? []
      const index = board.findIndex((c) => c.iid === e.iid)
      if (index >= 0) out.push({ seat: e.seat, card: board[index], index })
    }
    return out
  }

  /** Advance by one frame, computing fx for the forward transition. */
  function advance() {
    const prev = pb.current
    pb.next()
    const next = pb.current
    if (next.index === prev.index + 1 && next.seat !== null) {
      fx = computeFx(next.events, next.action, next.seat)
      dyingCards = dyingFrom(next.events, prev.snapshot)
      fxToken++
      pulse()
    } else {
      clearFx()
    }
    cursor = pb.cursor
  }

  function seek(i: number) { pb.seek(i); clearFx(); cursor = pb.cursor }
  function step(d: number) {
    if (d > 0) { advance() } else { pb.prev(); clearFx(); cursor = pb.cursor }
  }
  function turn(d: number) { d > 0 ? pb.nextTurn() : pb.prevTurn(); clearFx(); cursor = pb.cursor }
  function toggle() {
    playing = !playing
    if (playing) {
      timer = setInterval(() => {
        if (pb.cursor >= pb.frames.length - 1) { toggle(); return }
        advance()
      }, 600)
    } else if (timer) { clearInterval(timer); timer = null }
  }
  onDestroy(() => { if (timer) clearInterval(timer) })
</script>

<div class="viewer">
  <header>
    <button class="back" on:click={() => dispatch('back')}>← Library</button>
    <span class="title">
      <strong>{nameA}</strong> vs <strong>{nameB}</strong>
      · seed {replay.header.seed} · winner P{replay.header.winner} · {replay.header.turns} turns
    </span>
    <span class="tabs">
      <button class:on={tab === 'draft'} on:click={() => (tab = 'draft')}>Draft</button>
      <button class:on={tab === 'battle'} on:click={() => (tab = 'battle')}>Battle</button>
    </span>
  </header>

  {#if tab === 'draft'}
    <DraftPanel draft={replay.draft} />
  {:else}
    <div class="battle">
      <div class="gutter"></div>
      <div class="stage">
        <Board {snapshot} {nameA} {nameB} {fx} {fxToken} dying={dyingCards} />
        <Timeline {cursor} length={pb.frames.length} {playing}
          on:seek={(e) => seek(e.detail)} on:step={(e) => step(e.detail)}
          on:turn={(e) => turn(e.detail)} on:toggle={toggle} />
      </div>
      <div class="gutter right">
        <aside><ActionLog frames={pb.frames} {cursor} cardIds={pb.cardIds} on:seek={(e) => seek(e.detail)} /></aside>
      </div>
    </div>
  {/if}
</div>

<style>
  .viewer { color: #ddd; }
  header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 4px 6px 8px; font-size: 15px; }
  .title { color: #ccc; }
  .back { background: #23232b; color: #ddd; border: 1px solid #3a3f55; border-radius: 4px;
    padding: 3px 12px; cursor: pointer; font-weight: 600; }
  .back:hover { background: #2c2c38; }
  .tabs { margin-left: auto; display: flex; gap: 4px; }
  .tabs button { background: #23232b; color: #ddd; border: 1px solid #333;
    padding: 3px 12px; cursor: pointer; border-radius: 4px; }
  .tabs button.on { background: #2a2a44; }
  .battle { display: flex; gap: 16px; align-items: flex-start; }
  /* gutters share remaining width equally, so .stage stays centered in the view */
  .gutter { flex: 1 1 0; min-width: 0; display: flex; }
  .gutter.right { justify-content: flex-start; }
  .stage { flex: 0 0 auto; display: flex; flex-direction: column; gap: 4px; }
  /* action log: only as wide as its text */
  aside { width: max-content; max-width: 100%; }
</style>
