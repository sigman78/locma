<!-- web/src/components/Play/Play.svelte -->
<script lang="ts">
  import { onDestroy } from 'svelte'
  import { createGame, getGame, submitAction, submitDraft } from '../../lib/api'
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

  const END_DELAY_MS = 1800
  const HUMAN_FX_MS = 850

  // true only while the Play tab is the visible tab — the board's window-level
  // key handlers stay dormant when the tab is hidden but still mounted.
  export let active = true

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
  let finalBattle: BattlePending | null = null
  let showEnd = false
  let endTimer: ReturnType<typeof setTimeout> | null = null
  let staged: { cardIds: number[]; response: SubmitResponse } | null = null
  let lastOpponent: string | null = null

  loadCards()
    .then(() => (ready = true))
    .catch((e) => (error = String(e)))

  onDestroy(() => {
    seq?.cancel()
    if (endTimer) { clearTimeout(endTimer); endTimer = null }
    if (thinkTimer) { clearTimeout(thinkTimer); thinkTimer = null }
  })

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
          holdMs: 850,
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
      // playback finished — drop the live step + its lunge so the board settles on pending
      liveStep = null
      currentAction = null
    } else if (steps.length) {
      // a human move: render its result instantly. Keep currentAction set on THIS flush
      // so the attacker's lunge renders — resetting it here would batch to null before
      // Svelte flushes and the lunge would never play (spec §5).
      const s = steps[steps.length - 1]
      liveStep = null
      fire(s.events, s.action)
    } else {
      liveStep = null
      currentAction = null
    }
    if (r.result) {
      const last = steps[steps.length - 1]
      if (last) {
        finalBattle = { phase: 'battle', you, view: last.view, legal: [] }
        endTimer = setTimeout(
          () => (showEnd = true),
          paced ? END_DELAY_MS : HUMAN_FX_MS + END_DELAY_MS,
        )
      } else {
        // No board to freeze — reveal overlay immediately
        showEnd = true
      }
    }
    snap = { status: r.status, pending: r.pending, result: r.result }
  }

  async function start(detail: { opponent: string; seed?: number }) {
    try {
      lastOpponent = detail.opponent
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

  // In-flight request lock: `playing` only turns on AFTER a response arrives,
  // so without this a second click during a slow AI reply (vbeam thinking,
  // first-move model load) double-submits and the server answers 409 WrongPhase.
  let inFlight = false

  // "AI is thinking" hint: a search policy (vbeam/rbeam/netdmcts) can spend a
  // few seconds computing its turn. Show an animated indicator once a request
  // has been in flight past THINK_HINT_MS so a slow reply doesn't look hung.
  const THINK_HINT_MS = 1000
  let thinking = false
  let thinkTimer: ReturnType<typeof setTimeout> | null = null
  function armThinking() {
    thinkTimer = setTimeout(() => (thinking = true), THINK_HINT_MS)
  }
  function disarmThinking() {
    if (thinkTimer) { clearTimeout(thinkTimer); thinkTimer = null }
    thinking = false
  }

  async function pick(p: number) {
    if (!gameId || playing || inFlight) return
    inFlight = true
    armThinking()
    try {
      const r = await submitDraft(gameId, p)
      if (r.pending && r.pending.phase === 'battle') {
        staged = { cardIds: r.drafted ?? [], response: r }
      } else {
        await applyResponse(r, true)
      }
    } catch (e) {
      await recover(e)
    } finally {
      inFlight = false
      disarmThinking()
    }
  }

  // Auto-draft every remaining round with random picks; stage at the end.
  async function autoDraft() {
    if (!gameId || playing || inFlight) return
    inFlight = true
    armThinking()
    try {
      let r = await submitDraft(gameId, Math.floor(Math.random() * 3))
      while (r.pending && r.pending.phase === 'draft') {
        r = await submitDraft(gameId, Math.floor(Math.random() * 3))
      }
      staged = { cardIds: r.drafted ?? [], response: r }
    } catch (e) {
      await recover(e)
    } finally {
      inFlight = false
      disarmThinking()
    }
  }

  // A 409 means the action raced a phase change (double-submit, stale turn):
  // the game state is fine, so resync the snapshot instead of the error overlay.
  async function recover(e: unknown) {
    if (gameId && String(e).startsWith('Error: 409')) {
      try {
        snap = await getGame(gameId)
        return
      } catch {
        /* fall through to the overlay */
      }
    }
    error = String(e)
  }

  async function play() {
    if (!staged) return
    const s = staged
    staged = null
    // Pre-update snap to battle phase so the DraftScreen unmounts immediately and the
    // BattleScreen mounts before AI steps play — prevents draft cards re-flashing during
    // the AI step playback sequence (Fix 5). applyResponse sets snap again at the end
    // to the same values, which is harmless.
    snap = { status: s.response.status, pending: s.response.pending, result: s.response.result }
    await applyResponse(s.response, true)
  }

  async function act(a: ActionDict) {
    if (!gameId || playing || inFlight) return
    inFlight = true
    // only a turn-ending pass hands control to the AI; a plain board move
    // resolves on the server without the opponent searching, so skip the hint.
    if (a.t === 'pass') armThinking()
    try {
      await applyResponse(await submitAction(gameId, a), a.t === 'pass')
    } catch (e) {
      await recover(e)
    } finally {
      inFlight = false
      disarmThinking()
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
    finalBattle = null
    showEnd = false
    staged = null
    if (endTimer) { clearTimeout(endTimer); endTimer = null }
  }

  // Rematch: tear the finished game down, then immediately start a fresh one
  // against the same opponent (new random seed).
  async function rematch() {
    const opp = lastOpponent
    again()
    if (opp) await start({ opponent: opp })
  }

  $: battlePending = (snap?.pending && snap.pending.phase === 'battle')
    ? (snap.pending as BattlePending)
    : finalBattle
</script>

<main>
  {#if !ready}
    <p>loading cards…</p>
  {:else if !snap || !gameId}
    <h1>LOCM — Play vs AI</h1>
    <NewGame on:start={(e) => start(e.detail)} />
  {:else if snap.pending && snap.pending.phase === 'draft'}
    <DraftScreen
      {active}
      pending={snap.pending as DraftPending}
      done={!!staged}
      doneCardIds={staged?.cardIds ?? []}
      on:pick={(e) => pick(e.detail)}
      on:auto={autoDraft}
      on:play={play} />
  {:else if battlePending}
    <div class="board-stage">
      <BattleScreen
        {active}
        pending={battlePending}
        {you}
        {events}
        {currentAction}
        {fxToken}
        {liveStep}
        playing={playing || inFlight || !!snap?.result}
        on:act={(e) => act(e.detail)}
      />
      {#if thinking}
        <div class="thinking" role="status" aria-live="polite">
          <span class="spinner"></span>
          <span>AI is thinking<span class="dots"><i>.</i><i>.</i><i>.</i></span></span>
        </div>
      {/if}
      {#if showEnd && snap?.result}
        <EndOverlay {active} result={snap.result} opponent={lastOpponent} on:again={again} on:rematch={rematch} />
      {/if}
    </div>
  {:else if snap?.result}
    <EndOverlay {active} result={snap.result} opponent={lastOpponent} on:again={again} on:rematch={rematch} />
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
  /* the board is a fixed-size card layout; on narrow viewports let it pan
     horizontally instead of forcing a page-wide scrollbar */
  main { padding: 16px; color: #ddd; overflow-x: auto; }
  h1 { font-size: 20px; }
  .board-stage { position: relative; width: max-content; margin: 0 auto; }

  /* floating "AI is thinking" pill — only shown once a reply is slow, so the
     board never looks hung during a search policy's turn */
  .thinking { position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
    z-index: 40; display: flex; align-items: center; gap: 9px;
    padding: 7px 14px; border-radius: 999px; font-size: 13px; font-weight: 600;
    color: #cfd4f2; background: rgba(20, 20, 30, 0.92); border: 1px solid #3b3f6a;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5); animation: think-in 0.2s ease-out both; }
  @keyframes think-in { from { opacity: 0; transform: translate(-50%, -6px); }
    to { opacity: 1; transform: translate(-50%, 0); } }
  .spinner { width: 14px; height: 14px; border-radius: 50%;
    border: 2px solid #4a4f8a; border-top-color: #9aa0ff;
    animation: think-spin 0.7s linear infinite; }
  @keyframes think-spin { to { transform: rotate(360deg); } }
  .dots i { animation: think-blink 1.4s infinite both; }
  .dots i:nth-child(2) { animation-delay: 0.2s; }
  .dots i:nth-child(3) { animation-delay: 0.4s; }
  @keyframes think-blink { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }
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
