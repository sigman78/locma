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
  <h1>LOCM Replay Viewer</h1>
  {#if error}
    <p class="error">Error: {error}</p>
    <button on:click={() => (error = null)}>dismiss</button>
  {/if}
  {#if !ready && !error}
    <p>loading cards…</p>
  {:else if ready}
    {#if current}
      <button on:click={back}>← library</button>
      <ReplayViewer replay={current} />
    {:else}
      <ReplayLibrary on:open={(e) => open(e.detail)} />
    {/if}
  {/if}
</main>

<style>
  :global(body) { margin: 0; background: #0e0e12; font-family: system-ui, sans-serif; }
  main { width: 100%; margin: 0; padding: 16px; box-sizing: border-box; color: #ddd; }
  :global(#app) { width: 100%; max-width: none; }
  h1 { font-size: 18px; }
  .error { color: #ff6b6b; }
  button { background: #23232b; color: #ddd; border: 1px solid #333; border-radius: 4px;
    padding: 3px 10px; cursor: pointer; margin-bottom: 8px; }
</style>
