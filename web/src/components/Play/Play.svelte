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

  // auto-draft every remaining round with random picks (one server round-trip each).
  // We DON'T commit snap per pick: re-rendering the draft 30× would remount the card
  // images each round and the browser would cancel the still-in-flight /api/art
  // requests, which uvicorn logs as (harmless) WinError 10054 resets. Commit once at end.
  async function autoDraft() {
    if (!gameId) return
    try {
      let r = await submitDraft(gameId, Math.floor(Math.random() * 3))
      while (r.pending && r.pending.phase === 'draft') {
        r = await submitDraft(gameId, Math.floor(Math.random() * 3))
      }
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
  {#if !ready}
    <p>loading cards…</p>
  {:else if !snap || !gameId}
    <h1>LOCM — Play vs AI</h1>
    <NewGame on:start={(e) => start(e.detail)} />
  {:else if snap.result}
    <EndOverlay result={snap.result} on:again={again} />
  {:else if snap.pending && snap.pending.phase === 'draft'}
    <DraftScreen pending={snap.pending as DraftPending} on:pick={(e) => pick(e.detail)} on:auto={autoDraft} />
  {:else if snap.pending && snap.pending.phase === 'battle'}
    <BattleScreen pending={snap.pending as BattlePending} {you} {events} {fxToken} on:act={(e) => act(e.detail)} />
  {/if}

  <!-- blocking error overlay: a failed request leaves the game state unknown,
       so cover the UI and force a deliberate dismiss/reload rather than letting
       the user keep clicking a possibly-desynced board -->
  {#if error}
    <div class="error-overlay" role="alertdialog" aria-modal="true" aria-label="Connection error">
      <div class="error-box">
        <h2>Connection error</h2>
        <p>{error}</p>
        <div class="error-actions">
          <button on:click={() => (error = null)}>Dismiss</button>
          <button class="reload" on:click={() => location.reload()}>Reload</button>
        </div>
      </div>
    </div>
  {/if}
</main>

<style>
  :global(body) { margin: 0; background: #0e0e12; font-family: system-ui, sans-serif; }
  main { padding: 16px; color: #ddd; }
  h1 { font-size: 20px; }
  /* blocking modal: fixed full-viewport backdrop catches all clicks */
  .error-overlay { position: fixed; inset: 0; z-index: 1000;
    display: grid; place-items: center; background: rgba(0, 0, 0, 0.72); }
  .error-box { background: #1b1320; border: 1px solid #6a3a4f; border-radius: 10px;
    padding: 24px 28px; max-width: 460px; box-shadow: 0 14px 44px rgba(0, 0, 0, 0.7); }
  .error-box h2 { margin: 0 0 8px; color: #ff8a8a; font-size: 20px; }
  .error-box p { margin: 0; color: #e8c8c8; word-break: break-word; }
  .error-actions { display: flex; gap: 12px; margin-top: 18px; }
  .error-actions button { border-radius: 4px; padding: 8px 18px; cursor: pointer;
    font-weight: 600; background: #2a2230; color: #ddd; border: 1px solid #5a4250; }
  .error-actions .reload { background: #2a2a44; color: #fff; border-color: #4a4f6a; }
</style>
