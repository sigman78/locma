<!-- web/src/components/Replays/Replays.svelte — library + viewer (the old root app) -->
<script lang="ts">
  import ReplayLibrary from '../ReplayLibrary/ReplayLibrary.svelte'
  import ReplayViewer from '../ReplayViewer/ReplayViewer.svelte'
  import { getReplay } from '../../lib/api'
  import type { Replay } from '../../lib/replay'

  let current: Replay | null = null
  let error: string | null = null

  async function open(id: string) {
    try {
      current = await getReplay(id)
    } catch (e) {
      error = String(e)
    }
  }
</script>

<div>
  {#if error}
    <p class="error">Error: {error} <button on:click={() => (error = null)}>dismiss</button></p>
  {/if}
  {#if current}
    <ReplayViewer replay={current} on:back={() => (current = null)} />
  {:else}
    <ReplayLibrary on:open={(e) => open(e.detail)} />
  {/if}
</div>

<style>
  .error { color: #ff6b6b; padding: 0 12px; }
  button { background: #23232b; color: #ddd; border: 1px solid #333; border-radius: 4px;
    padding: 3px 10px; cursor: pointer; }
</style>
