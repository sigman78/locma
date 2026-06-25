<!-- web/src/components/Play/Play.svelte -->
<script lang="ts">
  import { createGame, submitAction, submitDraft } from '../../lib/api'
  import { loadCards } from '../../lib/cards'
  import type { ActionDict, EventDict } from '../../lib/replay'
  import type { BattlePending, CreatedGame, DraftPending, GameSnapshot } from '../../lib/play'
  import BattleScreen from './BattleScreen.svelte'
  import DraftScreen from './DraftScreen.svelte'
  import EndOverlay from './EndOverlay.svelte'
  import NewGame from './NewGame.svelte'

  let ready = false
  let error: string | null = null
  let gameId: string | null = null
  let you = 0
  let snap: GameSnapshot | null = null
  let events: EventDict[] = []
  let fxToken = 0

  loadCards().then(() => (ready = true)).catch((e) => (error = String(e)))

  async function start(detail: { opponent: string; seed?: number }) {
    try {
      const g: CreatedGame = await createGame({ opponent: detail.opponent, seed: detail.seed })
      gameId = g.game_id
      you = g.you
      snap = { status: g.status, pending: g.pending, result: g.result }
      events = []
    } catch (e) {
      error = String(e)
    }
  }

  async function pick(p: number) {
    if (!gameId) return
    try {
      const r = await submitDraft(gameId, p)
      events = r.slice.events
      fxToken++
      snap = { status: r.status, pending: r.pending, result: r.result }
    } catch (e) {
      error = String(e)
    }
  }

  async function act(a: ActionDict) {
    if (!gameId) return
    try {
      const r = await submitAction(gameId, a)
      events = r.slice.events
      fxToken++
      snap = { status: r.status, pending: r.pending, result: r.result }
    } catch (e) {
      error = String(e)
    }
  }

  function again() {
    gameId = null
    snap = null
    events = []
  }
</script>

<main>
  {#if error}
    <p class="error">Error: {error}</p>
    <button on:click={() => (error = null)}>dismiss</button>
  {/if}
  {#if !ready}
    <p>loading cards…</p>
  {:else if !snap || !gameId}
    <h1>LOCM — Play vs AI</h1>
    <NewGame on:start={(e) => start(e.detail)} />
  {:else if snap.result}
    <EndOverlay result={snap.result} on:again={again} />
  {:else if snap.pending && snap.pending.phase === 'draft'}
    <DraftScreen pending={snap.pending as DraftPending} on:pick={(e) => pick(e.detail)} />
  {:else if snap.pending && snap.pending.phase === 'battle'}
    <BattleScreen pending={snap.pending as BattlePending} {you} {events} {fxToken} on:act={(e) => act(e.detail)} />
  {/if}
</main>

<style>
  :global(body) { margin: 0; background: #0e0e12; font-family: system-ui, sans-serif; }
  main { padding: 16px; color: #ddd; }
  h1 { font-size: 20px; }
  .error { color: #ff6b6b; }
</style>
