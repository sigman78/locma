<!-- web/src/App.svelte -->
<script lang="ts">
  import ReplayLibrary from './components/ReplayLibrary/ReplayLibrary.svelte'
  import ReplayViewer from './components/ReplayViewer/ReplayViewer.svelte'
  import { getReplay } from './lib/api'
  import { loadCards } from './lib/cards'
  import type { Replay } from './lib/replay'

  let ready = false
  let current: Replay | null = null
  loadCards().then(() => (ready = true))

  async function open(id: string) { current = await getReplay(id) }
  function back() { current = null }
</script>

<main>
  <h1>LOCM Replay Viewer</h1>
  {#if !ready}
    <p>loading cards…</p>
  {:else if current}
    <button on:click={back}>← library</button>
    <ReplayViewer replay={current} />
  {:else}
    <ReplayLibrary on:open={(e) => open(e.detail)} />
  {/if}
</main>

<style>
  :global(body) { margin: 0; background: #0e0e12; font-family: system-ui, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px; color: #ddd; }
  h1 { font-size: 18px; }
  button { background: #23232b; color: #ddd; border: 1px solid #333; border-radius: 4px;
    padding: 3px 10px; cursor: pointer; margin-bottom: 8px; }
</style>
