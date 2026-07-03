<!-- Depot tab: artifact index with provenance, pin/pull/push, publish, gc. -->
<script lang="ts">
  import { onMount } from 'svelte'
  import {
    depotGc, depotPin, depotPublish, depotPull, depotPush, depotRemote, listDepot,
    type DepotRecord, type DepotVersion,
  } from '../../lib/api'

  export let active = true

  let records: DepotRecord[] = []
  let remote = ''
  let error: string | null = null
  let notice: string | null = null
  let busy: Record<string, boolean> = {}
  let open: Record<string, boolean> = {}

  // publish form
  let pubName = ''
  let pubFiles = ''
  let pubKind = 'model'
  let pubNote = ''
  let showPublish = false

  const human = (n: number) => {
    for (const u of ['B', 'KB', 'MB', 'GB']) {
      if (n < 1024 || u === 'GB') return u === 'B' ? `${n}B` : `${n.toFixed(1)}${u}`
      n /= 1024
    }
    return `${n}B`
  }

  async function refresh() {
    try {
      records = await listDepot()
      remote = (await depotRemote()).remote
    } catch (e) {
      error = String(e)
    }
  }
  onMount(refresh)
  $: if (active) refresh()

  async function act(key: string, fn: () => Promise<unknown>, done: (r: any) => string) {
    busy[key] = true
    error = notice = null
    try {
      notice = done(await fn())
      await refresh()
    } catch (e) {
      error = String(e)
    } finally { busy[key] = false }
  }

  const pull = (name: string) =>
    act(`pull-${name}`, () => depotPull(name), (r) =>
      r.fetched.length ? `pulled ${name}: ${r.fetched.join(', ')}` : `${name} already local`)
  const push = (name: string) =>
    act(`push-${name}`, () => depotPush(name), (r) => `pushed ${name} -> ${r.locator}`)
  const pin = (name: string, v: number) =>
    act(`pin-${name}`, () => depotPin(name, v), () => `pinned ${name} -> v${v}`)
  const gc = (dry: boolean) =>
    act('gc', () => depotGc(dry), (r) =>
      `${dry ? 'would remove' : 'removed'} ${r.removed} blob(s), ${human(r.freed)}`)

  const publish = () =>
    act('publish', () => depotPublish({
      name: pubName.trim(),
      files: pubFiles.split(/[\n,]/).map((s) => s.trim()).filter(Boolean),
      kind: pubKind,
      note: pubNote,
    }), (r) => {
      showPublish = false
      pubName = pubFiles = pubNote = ''
      return `published ${r.record.name} v${r.version}`
    })

  const pinned = (rec: DepotRecord): DepotVersion | undefined =>
    rec.versions.find((v) => v.version === rec.pin)
</script>

<div class="depot">
  <div class="topline">
    <span class="dim">remote: <code>{remote}</code></span>
    <span class="spacer"></span>
    <button on:click={() => (showPublish = !showPublish)}>publish…</button>
    <button on:click={() => gc(true)} disabled={busy['gc']}>gc (dry run)</button>
    <button class="danger" on:click={() => gc(false)} disabled={busy['gc']}>gc --yes</button>
  </div>

  {#if showPublish}
    <form class="pub" on:submit|preventDefault={publish}>
      <label>name <input bind:value={pubName} placeholder="my-net" required /></label>
      <label>kind
        <select bind:value={pubKind}>
          <option>model</option><option>dataset</option><option>eval</option><option>other</option>
        </select>
      </label>
      <label class="grow">files (server paths, comma or newline separated)
        <input bind:value={pubFiles} placeholder="runs/my_s0.zip, runs/my_s1.zip" required />
      </label>
      <label class="grow">note <input bind:value={pubNote} placeholder="what and why" /></label>
      <button class="primary" disabled={busy['publish']}>Publish</button>
    </form>
  {/if}

  {#if notice}<p class="notice">{notice}</p>{/if}
  {#if error}
    <p class="error">{error} <button on:click={() => (error = null)}>dismiss</button></p>
  {/if}

  <table>
    <thead>
      <tr>
        <th>name</th><th>kind</th><th>pin</th><th>versions</th><th>size@pin</th>
        <th>local</th><th>published</th><th></th>
      </tr>
    </thead>
    <tbody>
      {#each records as rec (rec.name)}
        {@const pv = pinned(rec)}
        <tr class="row" on:click={() => (open[rec.name] = !open[rec.name])}>
          <td class="mono name">{rec.name}</td>
          <td>{rec.kind}</td>
          <td>{rec.pin ? `v${rec.pin}` : '—'}</td>
          <td>{rec.versions.length}</td>
          <td>{pv ? human(pv.size) : '—'}</td>
          <td class:ok={pv?.status === 'local'} class:warn={pv?.status !== 'local'}>
            {pv?.status ?? '—'}
          </td>
          <td>{pv?.published ? 'yes' : 'no'}</td>
          <td class="acts" on:click|stopPropagation>
            <button on:click={() => pull(rec.name)} disabled={busy[`pull-${rec.name}`]}>
              pull
            </button>
            <button on:click={() => push(rec.name)} disabled={busy[`push-${rec.name}`]}>
              push
            </button>
          </td>
        </tr>
        {#if open[rec.name]}
          {#each [...rec.versions].reverse() as v (v.version)}
            <tr class="ver">
              <td colspan="8">
                <div class="verhead">
                  <strong>v{v.version}</strong>
                  {#if v.version === rec.pin}<span class="pinbadge">pin</span>
                  {:else}
                    <button class="mini" on:click={() => pin(rec.name, v.version)}>pin this</button>
                  {/if}
                  <span class="dim">{v.created}</span>
                  <span class="dim">commit {v.git_commit ?? '?'}{v.git_dirty ? '+dirty' : ''}</span>
                  <span class="dim">{human(v.size)} · {v.status}</span>
                  {#if v.published}<span class="dim mono">{v.published}</span>{/if}
                </div>
                {#if v.note}<div class="note">{v.note}</div>{/if}
                <div class="files mono">
                  {#each Object.entries(v.files) as [f, meta]}
                    <span title={meta.sha256}>{f}</span>
                  {/each}
                  {#if v.parents.length}
                    <span class="dim">parents: {v.parents.join(', ')}</span>
                  {/if}
                </div>
                {#if Object.keys(v.meta ?? {}).length}
                  <details>
                    <summary>meta</summary>
                    <pre>{JSON.stringify(v.meta, null, 2)}</pre>
                  </details>
                {/if}
              </td>
            </tr>
          {/each}
        {/if}
      {:else}
        <tr><td colspan="8" class="dim">depot is empty — publish something</td></tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .depot { font-size: 13px; }
  .topline { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }
  .spacer { flex: 1; }
  code { color: #9ab; }
  button { background: #1c1c24; border: 1px solid #2a2a36; color: #bbb; border-radius: 5px;
    padding: 5px 12px; cursor: pointer; font-size: 12px; }
  button:hover { color: #fff; }
  button:disabled { opacity: 0.5; cursor: default; }
  button.danger:hover { color: #ff6b6b; border-color: #ff6b6b; }
  button.primary { background: #234a2c; border-color: #3fbf66; color: #d7ffd7; }
  .pub { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap;
    background: #14141a; border: 1px solid #23232b; border-radius: 8px; padding: 12px 14px;
    margin-bottom: 10px; }
  .pub label { display: flex; flex-direction: column; gap: 4px; color: #9a9ab0; }
  .pub .grow { flex: 1; min-width: 260px; }
  .pub input, .pub select { background: #23232b; color: #ddd; border: 1px solid #3a3f55;
    border-radius: 4px; padding: 6px 10px; font-size: 13px; }
  .notice { color: #4fd97a; }
  .error { color: #ff6b6b; }
  .error button { margin-left: 6px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #1e1e26; }
  th { color: #777; font-weight: 500; font-size: 12px; }
  .row { cursor: pointer; }
  .row:hover { background: #16161e; }
  .name { color: #eee; font-weight: 600; }
  .mono { font-family: ui-monospace, Consolas, monospace; }
  .ok { color: #4fd97a; }
  .warn { color: #d9a64f; }
  .acts { white-space: nowrap; }
  .acts button { margin-right: 4px; }
  .ver td { background: #101016; }
  .verhead { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
  .pinbadge { background: #23233a; border: 1px solid #4a4f8a; color: #aab;
    border-radius: 4px; padding: 1px 8px; font-size: 11px; }
  .mini { padding: 2px 8px; font-size: 11px; }
  .note { color: #aaa; margin: 6px 0 0; max-width: 900px; }
  .files { display: flex; gap: 14px; margin-top: 6px; color: #8aa; flex-wrap: wrap; }
  .dim { color: #777; }
  details { margin-top: 6px; }
  summary { color: #666; cursor: pointer; font-size: 12px; }
  pre { color: #999; font-size: 12px; }
</style>
