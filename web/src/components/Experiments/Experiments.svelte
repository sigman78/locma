<!-- Experiments tab: preset library, schema-driven editor, background jobs
     with live progress, and per-kind result views. -->
<script lang="ts">
  import { onDestroy, onMount } from 'svelte'
  import {
    cancelJob, deletePreset, getExperimentKinds, getJobLog, getJobSeries, listJobs,
    listPresets, runExperiment, savePreset,
    type ExperimentKind, type ExpJob, type JobSeries, type Preset,
  } from '../../lib/api'
  import JobCharts from './JobCharts.svelte'
  import ParamsForm from './ParamsForm.svelte'
  import ResultView from './ResultView.svelte'

  export let active = true

  let kinds: ExperimentKind[] = []
  let presets: Preset[] = []
  let jobs: ExpJob[] = []
  let error: string | null = null
  let busy = false

  // editor state
  let kind = 'match'
  let name = ''
  let note = ''
  let params: Record<string, any> = {}
  let loadedPresetId: string | null = null

  let expanded: Record<string, boolean> = {}
  let seriesMap: Record<string, JobSeries> = {}
  let logMap: Record<string, string | null> = {} // null = show panel, still loading
  let timer: ReturnType<typeof setInterval> | null = null

  $: currentKind = kinds.find((k) => k.kind === kind)

  const defaults = (k: ExperimentKind): Record<string, any> =>
    Object.fromEntries(k.schema.map((f) => [
      f.name,
      Array.isArray(f.default) ? [...(f.default as string[])] : f.default,
    ]))

  function pickKind(id: string) {
    kind = id
    const k = kinds.find((x) => x.kind === id)
    if (k) params = defaults(k)
    loadedPresetId = null
  }

  function loadPreset(p: Preset) {
    kind = p.kind
    name = p.name
    note = p.note ?? ''
    const k = kinds.find((x) => x.kind === p.kind)
    params = { ...(k ? defaults(k) : {}), ...structuredClone(p.params) }
    loadedPresetId = p.id
  }

  const slug = (s: string) =>
    s.toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || 'preset'

  async function fetchSeries(id: string) {
    try {
      seriesMap[id] = await getJobSeries(id)
    } catch {
      /* series may not exist yet */
    }
  }

  const ACTIVE = new Set(['queued', 'running'])

  async function refreshJobs() {
    try {
      const prev = Object.fromEntries(jobs.map((j) => [j.job_id, j.state]))
      jobs = await listJobs()
      for (const j of jobs) {
        // refresh curves for expanded jobs while they run, and once more on finish
        const wasActive = ACTIVE.has(prev[j.job_id] ?? '')
        if (expanded[j.job_id] && (ACTIVE.has(j.state) || (wasActive && !ACTIVE.has(j.state)))) {
          fetchSeries(j.job_id)
        }
      }
    } catch (e) {
      error = String(e)
    }
  }

  function toggle(id: string) {
    expanded[id] = !expanded[id]
    if (expanded[id] && !seriesMap[id]) fetchSeries(id)
  }

  async function showLog(id: string) {
    if (logMap[id] !== undefined) { delete logMap[id]; logMap = logMap; return } // toggle off
    logMap[id] = null
    try {
      logMap[id] = (await getJobLog(id)).log || '(log is empty)'
    } catch (e) {
      logMap[id] = String(e)
    }
  }

  async function save() {
    if (!name.trim()) { error = 'preset needs a name'; return }
    busy = true
    try {
      const id = loadedPresetId ?? slug(name)
      await savePreset(id, { name, kind, params, note })
      loadedPresetId = id
      presets = await listPresets()
      error = null
    } catch (e) {
      error = String(e)
    } finally { busy = false }
  }

  async function remove(p: Preset) {
    busy = true
    try {
      await deletePreset(p.id)
      if (loadedPresetId === p.id) loadedPresetId = null
      presets = await listPresets()
    } catch (e) {
      error = String(e)
    } finally { busy = false }
  }

  async function run() {
    busy = true
    try {
      const job = await runExperiment({ kind, params, name: name || undefined })
      expanded[job.job_id] = true
      await refreshJobs()
      error = null
    } catch (e) {
      error = String(e)
    } finally { busy = false }
  }

  async function cancel(id: string) {
    try {
      await cancelJob(id)
      await refreshJobs()
    } catch (e) {
      error = String(e)
    }
  }

  $: anyRunning = jobs.some((j) => j.state === 'queued' || j.state === 'running')

  onMount(async () => {
    try {
      kinds = await getExperimentKinds()
      if (kinds.length) pickKind(kinds[0].kind)
      presets = await listPresets()
      await refreshJobs()
    } catch (e) {
      error = String(e)
    }
    timer = setInterval(() => {
      if (active && anyRunning) refreshJobs()
    }, 1200)
  })
  onDestroy(() => { if (timer) clearInterval(timer) })

  const dur = (j: ExpJob) => {
    if (!j.started) return ''
    const s = (j.finished ?? Date.now() / 1000) - j.started
    return s < 90 ? `${s.toFixed(1)}s` : `${(s / 60).toFixed(1)}m`
  }
  const stateColor: Record<string, string> = {
    running: '#4fa3d9', queued: '#888', done: '#4fd97a', error: '#ff6b6b', cancelled: '#d9a64f',
  }
</script>

<div class="wrap">
  <aside>
    <h3>Presets</h3>
    {#each presets as p}
      <div class="preset" class:sel={p.id === loadedPresetId}>
        <button class="load" on:click={() => loadPreset(p)} title={p.note}>
          <span class="pname">{p.name}</span>
          <span class="pkind">{p.kind}</span>
        </button>
        <button class="del" on:click={() => remove(p)} title="delete preset">x</button>
      </div>
    {:else}
      <p class="dim">none saved yet</p>
    {/each}
  </aside>

  <section class="editor">
    <div class="kindbar">
      {#each kinds as k}
        <button class:active={kind === k.kind} on:click={() => pickKind(k.kind)} title={k.description}>
          {k.label}
        </button>
      {/each}
    </div>
    {#if currentKind}
      <p class="dim desc">{currentKind.description}</p>
      <div class="meta">
        <label>name <input bind:value={name} placeholder="my experiment" /></label>
        <label>note <input bind:value={note} placeholder="optional" class="wide" /></label>
      </div>
      <ParamsForm schema={currentKind.schema} bind:params />
      <div class="actions">
        <button class="primary" on:click={run} disabled={busy}>Run</button>
        <button on:click={save} disabled={busy}>
          {loadedPresetId ? 'Save preset' : 'Save as preset'}
        </button>
        {#if loadedPresetId}
          <button on:click={() => { loadedPresetId = null }} disabled={busy}>Save as copy…</button>
        {/if}
      </div>
    {/if}
    {#if error}
      <p class="error">{error} <button on:click={() => (error = null)}>dismiss</button></p>
    {/if}
  </section>
</div>

<section class="jobs">
  <h3>Runs</h3>
  {#each jobs as j (j.job_id)}
    <div class="job">
      <button class="jobhead" on:click={() => toggle(j.job_id)}>
        <span class="state" style="color: {stateColor[j.state] ?? '#888'}">{j.state}</span>
        <span class="jname">{j.name}</span>
        <span class="jkind">{j.kind}</span>
        <span class="prog">
          {#if j.state === 'running' || j.state === 'queued'}
            <span class="bar"><span
              class="fill"
              style="width: {j.progress_total ? (100 * j.progress_done) / j.progress_total : 0}%"
            ></span></span>
            {j.progress_done}/{j.progress_total}
          {:else}
            {dur(j)}
          {/if}
        </span>
      </button>
      {#if j.state === 'running' || j.state === 'queued'}
        <button class="cancel" on:click={() => cancel(j.job_id)}>cancel</button>
      {/if}
      {#if expanded[j.job_id]}
        <div class="detail">
          {#if seriesMap[j.job_id]}
            <JobCharts
              kind={j.kind}
              series={seriesMap[j.job_id].series}
              live={seriesMap[j.job_id].live}
            />
          {/if}
          {#if j.result}
            <ResultView kind={j.kind} result={j.result} />
          {:else if j.error}
            <p class="error">{j.error}</p>
          {:else}
            <p class="dim">running…</p>
          {/if}
          <div class="detailrow">
            <details>
              <summary>params</summary>
              <pre>{JSON.stringify(j.params, null, 2)}</pre>
            </details>
            <button class="loglink" on:click={() => showLog(j.job_id)}>
              {logMap[j.job_id] === undefined ? 'view log' : 'hide log'}
            </button>
          </div>
          {#if logMap[j.job_id] !== undefined}
            <pre class="log">{logMap[j.job_id] ?? 'loading…'}</pre>
          {/if}
        </div>
      {/if}
    </div>
  {:else}
    <p class="dim">no runs yet — configure an experiment above and hit Run</p>
  {/each}
</section>

<style>
  .wrap { display: flex; gap: 20px; align-items: flex-start; }
  aside { width: 220px; flex-shrink: 0; }
  h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #777;
    margin: 4px 0 8px; }
  .preset { display: flex; align-items: stretch; gap: 4px; margin-bottom: 4px; }
  .preset .load { flex: 1; display: flex; justify-content: space-between; gap: 8px;
    background: #16161d; border: 1px solid #26262f; color: #ccc; border-radius: 6px;
    padding: 7px 10px; cursor: pointer; text-align: left; font-size: 13px; }
  .preset.sel .load { border-color: #4a4f8a; background: #1a1a2c; }
  .preset .load:hover { border-color: #3a3a4a; }
  .pkind { color: #666; font-size: 11px; }
  .del { background: none; border: none; color: #555; cursor: pointer; }
  .del:hover { color: #ff6b6b; }

  .editor { flex: 1; background: #14141a; border: 1px solid #23232b; border-radius: 8px;
    padding: 14px 16px; }
  .kindbar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
  .kindbar button { background: #1c1c24; border: 1px solid #2a2a36; color: #aaa;
    border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 13px; }
  .kindbar button.active { background: #23233a; color: #fff; border-color: #4a4f8a; }
  .desc { margin: 2px 0 10px; }
  .meta { display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }
  .meta label { display: flex; gap: 6px; align-items: center; font-size: 13px; color: #9a9ab0; }
  .meta input { background: #23232b; color: #ddd; border: 1px solid #3a3f55; border-radius: 4px;
    padding: 6px 10px; font-size: 13px; }
  .meta .wide { width: 280px; }
  .actions { display: flex; gap: 10px; margin-top: 14px; }
  .actions button { background: #1c1c24; border: 1px solid #2a2a36; color: #bbb;
    border-radius: 6px; padding: 7px 16px; cursor: pointer; font-size: 13px; }
  .actions .primary { background: #234a2c; border-color: #3fbf66; color: #d7ffd7;
    font-weight: 600; }
  .actions button:disabled { opacity: 0.5; cursor: default; }

  .jobs { margin-top: 20px; }
  .job { position: relative; margin-bottom: 6px; }
  .jobhead { width: 100%; display: flex; gap: 14px; align-items: center;
    background: #14141a; border: 1px solid #23232b; border-radius: 6px; padding: 8px 12px;
    color: #ccc; cursor: pointer; font-size: 13px; text-align: left; }
  .jobhead:hover { border-color: #33333f; }
  .state { width: 70px; font-weight: 600; flex-shrink: 0; }
  .jname { flex: 1; }
  .jkind { color: #666; }
  .prog { display: flex; align-items: center; gap: 8px; color: #888; min-width: 130px;
    justify-content: flex-end; }
  .bar { width: 90px; height: 6px; background: #23232b; border-radius: 3px; overflow: hidden; }
  .fill { display: block; height: 100%; background: #4fa3d9; transition: width 0.4s ease; }
  .cancel { position: absolute; right: 150px; top: 7px; background: none; border: 1px solid #444;
    color: #999; border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 11px; }
  .cancel:hover { color: #ff6b6b; border-color: #ff6b6b; }
  .detail { border: 1px solid #23232b; border-top: none; border-radius: 0 0 6px 6px;
    padding: 12px 14px; background: #101016; display: flex; flex-direction: column;
    gap: 10px; }
  .detailrow { display: flex; gap: 16px; align-items: flex-start; }
  details { margin-top: 0; flex: 1; }
  summary { color: #666; font-size: 12px; cursor: pointer; }
  pre { font-size: 12px; color: #999; }
  .loglink { background: none; border: 1px solid #333; color: #888; border-radius: 4px;
    padding: 2px 10px; cursor: pointer; font-size: 11px; }
  .loglink:hover { color: #ddd; }
  .log { background: #0c0c10; border: 1px solid #1c1c24; border-radius: 6px;
    padding: 10px; max-height: 320px; overflow: auto; white-space: pre-wrap; }
  .dim { color: #777; font-size: 13px; }
  .error { color: #ff6b6b; font-size: 13px; }
  .error button { background: none; border: 1px solid #444; color: #999; border-radius: 4px;
    padding: 1px 8px; cursor: pointer; margin-left: 6px; }
</style>
