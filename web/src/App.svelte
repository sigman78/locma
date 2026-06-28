<script lang="ts">
  import ReplayLibrary from './components/ReplayLibrary/ReplayLibrary.svelte'
  import ReplayViewer from './components/ReplayViewer/ReplayViewer.svelte'
  import { getReplay } from './lib/api'
  import { loadCards } from './lib/cards'
  import type { Replay } from './lib/replay'

  let ready = false
  let error: string | null = null
  let current: Replay | null = null

  loadCards().then(() => (ready = true)).catch((e) => (error = String(e)))

  async function open(id: string) {
    try {
      current = await getReplay(id)
    } catch (e) {
      error = String(e)
    }
  }
  function back() { current = null }
</script>

<main>
  {#if error}
    <p class="error">Error: {error}</p>
    <button on:click={() => (error = null)}>dismiss</button>
  {/if}
  {#if !ready && !error}
    <p>loading cards…</p>
  {:else if ready}
    {#if current}
      <ReplayViewer replay={current} on:back={back} />
    {:else}
      <div class="topbar">
        <h1>LOCM Replay Viewer</h1>
        <a class="playlink" href="game.html">🎮 Play vs AI</a>
      </div>
      <ReplayLibrary on:open={(e) => open(e.detail)} />
    {/if}
  {/if}
</main>

<style>
  :global(body) { margin: 0; background: #0e0e12; font-family: system-ui, sans-serif; }
  main { width: 100%; margin: 0; padding: 16px; box-sizing: border-box; color: #ddd; }
  :global(#app) { width: 100%; max-width: none; }
  h1 { font-size: 18px; }
  .topbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .playlink { text-decoration: none; font-weight: 600; font-size: 14px; color: #0e0e12;
    background: #4fd97a; border: 1px solid #3fbf66; border-radius: 6px; padding: 7px 14px;
    box-shadow: 0 0 10px rgba(79, 217, 122, 0.35); transition: filter 0.12s ease, transform 0.12s ease; }
  .playlink:hover { filter: brightness(1.08); transform: translateY(-1px); }
  .error { color: #ff6b6b; }
  button { background: #23232b; color: #ddd; border: 1px solid #333; border-radius: 4px;
    padding: 3px 10px; cursor: pointer; margin-bottom: 8px; }
</style>
