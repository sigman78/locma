<!-- web/src/components/Play/EndOverlay.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import type { GameResult } from '../../lib/play'

  export let result: GameResult
  const dispatch = createEventDispatcher<{ again: void }>()
</script>

<div class="overlay">
  <div class="card">
    <h1 class:win={result.winner_is_human}>{result.winner_is_human ? 'You win! 🎉' : 'You lose'}</h1>
    <p>{result.turns} turns</p>
    <p class="rid">replay {result.replay_id}</p>
    <div class="actions">
      <button on:click={() => dispatch('again')}>Try again</button>
      <a href="/index.html" class="replay-btn">Replay</a>
    </div>
  </div>
</div>

<style>
  .overlay {
    position: absolute;
    inset: 0;
    z-index: 50;
    background: rgba(8, 8, 12, 0.78);
    display: grid;
    place-items: center;
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: 12px;
    align-items: center;
    background: #15151b;
    border: 1px solid #2a2a36;
    border-radius: 10px;
    padding: 40px;
    color: #ddd;
    animation: pop-in 0.25s ease-out both;
  }
  @keyframes pop-in {
    from { opacity: 0; transform: scale(0.88); }
    to   { opacity: 1; transform: scale(1); }
  }
  h1 { font-size: 34px; color: #ff8a8a; margin: 0; }
  h1.win { color: #7ddf7d; }
  p { margin: 0; }
  .rid { color: #777; font-size: 12px; }
  .actions { display: flex; gap: 12px; margin-top: 8px; }
  button, .replay-btn {
    background: #2a2a44;
    color: #fff;
    border: 1px solid #4a4f6a;
    border-radius: 4px;
    padding: 8px 18px;
    cursor: pointer;
    font-weight: 600;
    text-decoration: none;
    font-size: inherit;
    display: inline-block;
  }
</style>
