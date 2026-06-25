<!-- web/src/components/Play/EndOverlay.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import type { GameResult } from '../../lib/play'

  export let result: GameResult
  const dispatch = createEventDispatcher<{ again: void }>()
</script>

<div class="overlay">
  <h1 class:win={result.winner_is_human}>{result.winner_is_human ? 'You win! 🎉' : 'You lose'}</h1>
  <p>{result.turns} turns</p>
  <p class="links">
    <a href="/index.html">View replays →</a>
    <span class="rid">replay {result.replay_id}</span>
  </p>
  <button on:click={() => dispatch('again')}>Play again</button>
</div>

<style>
  .overlay { display: flex; flex-direction: column; gap: 12px; align-items: center;
    background: #15151b; border: 1px solid #2a2a36; border-radius: 10px; padding: 40px; color: #ddd; }
  h1 { font-size: 34px; color: #ff8a8a; }
  h1.win { color: #7ddf7d; }
  .links { display: flex; gap: 16px; align-items: center; }
  .rid { color: #777; font-size: 12px; }
  a { color: #6bb8ff; }
  button { background: #2a2a44; color: #fff; border: 1px solid #4a4f6a; border-radius: 4px;
    padding: 8px 18px; cursor: pointer; font-weight: 600; }
</style>
