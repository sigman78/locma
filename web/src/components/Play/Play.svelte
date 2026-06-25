<!-- web/src/components/Play/Play.svelte -->
<script lang="ts">
  import { onDestroy } from 'svelte'
  import { createGame, submitAction, submitDraft } from '../../lib/api'
  import { loadCards } from '../../lib/cards'
  import type { ActionDict, EventDict } from '../../lib/replay'
  import type {
    BattlePending,
    CreatedGame,
    DraftPending,
    GameSnapshot,
    PlayStep,
    SubmitResponse,
  } from '../../lib/play'
  import { playFrames, type Sequencer } from '../../lib/playback'
  import { pulse } from '../../lib/motion'
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
  let currentAction: ActionDict | null = null
  let fxToken = 0
  let liveStep: PlayStep | null = null
  let playing = false
  let seq: Sequencer | null = null

  loadCards()
    .then(() => (ready = true))
    .catch((e) => (error = String(e)))

  onDestroy(() => seq?.cancel())

  function fire(evs: EventDict[], action: ActionDict | null) {
    events = evs
    currentAction = action
    fxToken++
    pulse(700)
  }

  // Animate the AI's ordered steps, then resolve. Locks input while playing.
  function playSequence(steps: PlayStep[]): Promise<void> {
    return new Promise((resolve) => {
      playing = true
      seq = playFrames(
        steps,
        (s) => {
          liveStep = s
          fire(s.events, s.action)
        },
        {
          holdMs: 650,
          onDone: () => {
            playing = false
            seq = null
            resolve()
          },
        },
      )
    })
  }

  // `paced` (End Turn / draft→battle) animates the AI steps; a human battle move
  // renders its single resulting step instantly.
  async function applyResponse(r: SubmitResponse, paced: boolean) {
    const steps = r.steps ?? []
    if (paced && steps.length) {
      await playSequence(steps)
    } else if (steps.length) {
      const s = steps[steps.length - 1]
      fire(s.events, s.action)
    }
    liveStep = null
    currentAction = null
    snap = { status: r.status, pending: r.pending, result: r.result }
  }

  async function start(detail: { opponent: string; seed?: number }) {
    try {
      const g: CreatedGame = await createGame({ opponent: detail.opponent, seed: detail.seed })
      gameId = g.game_id
      you = g.you
      snap = { status: g.status, pending: g.pending, result: g.result }
      events = []
      currentAction = null
    } catch (e) {
      error = String(e)
    }
  }

  async function pick(p: number) {
    if (!gameId || playing) return
    try {
      await applyResponse(await submitDraft(gameId, p), true)
    } catch (e) {
      error = String(e)
    }
  }

  // Auto-draft every remaining round with random picks; commit once at the end.
  async function autoDraft() {
    if (!gameId || playing) return
    try {
      let r = await submitDraft(gameId, Math.floor(Math.random() * 3))
      while (r.pending && r.pending.phase === 'draft') {
        r = await submitDraft(gameId, Math.floor(Math.random() * 3))
      }
      await applyResponse(r, true)
    } catch (e) {
      error = String(e)
    }
  }

  async function act(a: ActionDict) {
    if (!gameId || playing) return
    try {
      await applyResponse(await submitAction(gameId, a), a.t === 'pass')
    } catch (e) {
      error = String(e)
    }
  }

  function again() {
    seq?.cancel()
    seq = null
    playing = false
    liveStep = null
    gameId = null
    snap = null
    events = []
    currentAction = null
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
    <BattleScreen
      pending={snap.pending as BattlePending}
      {you}
      {events}
      {currentAction}
      {fxToken}
      {liveStep}
      {playing}
      on:act={(e) => act(e.detail)}
    />
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
    font-weight: 600; background: #2a2a44; color: #fff; border: 1px solid #4a4f6a; }
</style>
