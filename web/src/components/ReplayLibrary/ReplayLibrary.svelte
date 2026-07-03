<!-- web/src/components/ReplayLibrary/ReplayLibrary.svelte -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte'
  import { importReplay, listGameLogs, listReplays, runReplay } from '../../lib/api'
  import type { ReplayHeader } from '../../lib/replay'
  import PolicyInput from '../shared/PolicyInput.svelte'

  const dispatch = createEventDispatcher<{ open: string }>()
  let rows: ReplayHeader[] = []
  let logs: { path: string; rows: number }[] = []
  let pa = 'greedy', pb = 'random', seed = 0
  let logPath = '', logRow = 0
  let busy = false
  let error: string | null = null

  async function refresh() {
    rows = await listReplays()
  }
  onMount(async () => {
    logs = await listGameLogs()
    await refresh()
  })

  async function run() {
    busy = true
    error = null
    try {
      const h = await runReplay({ policy_a: pa, policy_b: pb, seed })
      await refresh()
      dispatch('open', h.replay_id)
    } catch (e) {
      error = String(e)
    } finally { busy = false }
  }
  async function doImport() {
    if (!logPath) return
    busy = true
    error = null
    try {
      const h = await importReplay({ path: logPath, row: logRow })
      await refresh()
      dispatch('open', h.replay_id)
    } catch (e) {
      error = String(e)
    } finally { busy = false }
  }
</script>

<div class="lib">
  <section class="forms">
    <form on:submit|preventDefault={run}>
      <h3>Run a matchup</h3>
      <PolicyInput bind:value={pa} />
      vs
      <PolicyInput bind:value={pb} />
      seed <input type="number" bind:value={seed} style="width:60px" />
      <button disabled={busy}>Run</button>
    </form>
    <form on:submit|preventDefault={doImport}>
      <h3>Import from game-log</h3>
      <select bind:value={logPath}>
        <option value="">— select —</option>
        {#each logs as l}<option value={l.path}>{l.path} ({l.rows})</option>{/each}
      </select>
      row <input type="number" bind:value={logRow} style="width:60px" />
      <button disabled={busy || !logPath}>Import</button>
    </form>
  </section>
  {#if error}<p class="error">{error}</p>{/if}

  <table>
    <thead><tr><th>created</th><th>matchup</th><th>seed</th><th>winner</th><th>turns</th><th>source</th></tr></thead>
    <tbody>
      {#each rows as r}
        <tr on:click={() => dispatch('open', r.replay_id)}>
          <td>{r.created_at.slice(0, 19).replace('T', ' ')}</td>
          <td>{r.policy_a} vs {r.policy_b}</td>
          <td>{r.seed}</td><td>P{r.winner}</td><td>{r.turns}</td><td>{r.source}</td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .lib { color: #ddd; padding: 12px; }
  .forms { display: flex; gap: 24px; margin-bottom: 16px; }
  h3 { margin: 0 0 6px; font-size: 13px; color: #aaa; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #222; }
  tbody tr { cursor: pointer; } tbody tr:hover { background: #1c1c22; }
  select, input, button { background: #23232b; color: #ddd; border: 1px solid #333; }
  .error { color: #ff6b6b; font-size: 13px; }
</style>
