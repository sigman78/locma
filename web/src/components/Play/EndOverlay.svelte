<!-- web/src/components/Play/EndOverlay.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import type { GameResult } from '../../lib/play'

  export let result: GameResult
  // The opponent spec of the game just finished, so we can offer a one-click
  // rematch against the same AI. Absent -> only the "new opponent" path shows.
  export let opponent: string | null = null

  const dispatch = createEventDispatcher<{ again: void; rematch: void }>()
  const won = result.winner_is_human
</script>

<div class="overlay" role="dialog" aria-modal="true" aria-label="Game over">
  <div class="card" class:win={won}>
    <span class="badge" class:win={won}>{won ? 'Victory' : 'Defeat'}</span>
    <h1 class:win={won}>{won ? 'You win' : 'You lose'}</h1>
    <dl class="stats">
      <div><dt>Turns</dt><dd>{result.turns}</dd></div>
      <div><dt>Result</dt><dd>{won ? 'human' : 'AI'} wins</dd></div>
    </dl>
    <div class="actions">
      {#if opponent}
        <button class="primary" on:click={() => dispatch('rematch')}>Rematch</button>
        <button on:click={() => dispatch('again')}>New opponent</button>
      {:else}
        <button class="primary" on:click={() => dispatch('again')}>Play again</button>
      {/if}
      <a href="#/replays" class="replay-btn">View replay</a>
    </div>
    <p class="rid">replay {result.replay_id}</p>
  </div>
</div>

<style>
  .overlay {
    position: absolute;
    inset: 0;
    z-index: 50;
    background: rgba(8, 8, 12, 0.62);
    display: grid;
    place-items: center;
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: 10px;
    align-items: center;
    min-width: 300px;
    background: #15151b;
    border: 1px solid #47324a;
    border-radius: 12px;
    padding: 32px 40px;
    color: #ddd;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.6);
    animation: pop-in 0.25s ease-out both;
  }
  .card.win { border-color: #2f5a37; }
  @keyframes pop-in {
    from { opacity: 0; transform: scale(0.88); }
    to   { opacity: 1; transform: scale(1); }
  }
  .badge { font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
    font-weight: 700; color: #ff9a9a; background: rgba(255, 107, 107, 0.12);
    border: 1px solid #6a3a4f; border-radius: 999px; padding: 3px 12px; }
  .badge.win { color: #9be79b; background: rgba(79, 217, 122, 0.12); border-color: #2f5a37; }
  h1 { font-size: 34px; color: #ff8a8a; margin: 2px 0 4px; }
  h1.win { color: #7ddf7d; }
  .stats { display: flex; gap: 28px; margin: 4px 0 8px; }
  .stats div { display: flex; flex-direction: column; align-items: center; gap: 2px; }
  .stats dt { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #777; }
  .stats dd { margin: 0; font-size: 18px; font-weight: 600; color: #e6e6ee; }
  .rid { color: #666; font-size: 12px; margin: 2px 0 0; }
  .actions { display: flex; gap: 12px; margin-top: 6px; align-items: center; flex-wrap: wrap;
    justify-content: center; }
  button, .replay-btn {
    background: #2a2a44;
    color: #fff;
    border: 1px solid #4a4f6a;
    border-radius: 6px;
    padding: 9px 18px;
    cursor: pointer;
    font-weight: 600;
    text-decoration: none;
    font-size: inherit;
    display: inline-flex;
    align-items: center;
  }
  button.primary { background: #234a2c; border-color: #3fbf66; color: #d7ffd7; }
  button:hover, .replay-btn:hover { filter: brightness(1.15); }
</style>
