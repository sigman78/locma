<!-- User-friendly policy selector: curated dropdown (baselines, search
     presets, depot-backed model specs) + a "Custom spec…" free-text escape
     hatch, with an explanation of the selected spec's parameters underneath
     (hidden in compact mode). Reusable anywhere a single policy is chosen. -->
<script lang="ts">
  import { onMount } from 'svelte'
  import { policyCatalog } from '../../lib/catalog'
  import { explainSpec } from '../../lib/specs'
  import PolicyInput from './PolicyInput.svelte'

  export let value = ''
  export let compact = false
  /** Pick the strongest depot-backed spec as the initial value (Play mode). */
  export let defaultCompetitive = false

  const CUSTOM = '__custom__'
  interface Group {
    label: string
    options: { value: string; label: string }[]
  }
  let groups: Group[] = [
    { label: 'Baselines', options: [
      { value: 'greedy', label: 'Greedy' },
      { value: 'scripted', label: 'Scripted' },
      { value: 'max-guard', label: 'Max-Guard' },
      { value: 'max-attack', label: 'Max-Attack' },
      { value: 'random', label: 'Random' },
    ] },
    { label: 'Search', options: [
      { value: 'dmcts:15,30', label: 'DMCTS (fair search)' },
      { value: 'mcts:100', label: 'MCTS (cheating search)' },
      { value: 'azlite:100', label: 'AZ-lite (PUCT)' },
    ] },
  ]
  let custom = false

  $: allValues = groups.flatMap((g) => g.options.map((o) => o.value))
  $: selectValue = custom ? CUSTOM : allValues.includes(value) ? value : value ? CUSTOM : ''
  $: explanation = value ? explainSpec(value) : null

  function onSelect(e: Event) {
    const v = (e.currentTarget as HTMLSelectElement).value
    if (v === CUSTOM) {
      custom = true
    } else {
      custom = false
      value = v
    }
  }

  onMount(async () => {
    try {
      const cat = await policyCatalog()
      const models: Group['options'] = []
      for (const m of cat.depot_models) {
        const ref = m.refs[0]
        if (!ref) continue
        models.push({ value: `vbeam:${ref}`, label: `V-beam · ${m.name} (strongest)` })
        models.push({ value: `ppo:${ref}`, label: `PPO · ${m.name} (reactive)` })
      }
      if (models.length) groups = [{ label: 'Models (depot)', options: models }, ...groups]
      if (defaultCompetitive && !value) {
        const b0 = cat.depot_models.find((m) => m.name === 'b0') ?? cat.depot_models[0]
        value = b0?.refs[0] ? `vbeam:${b0.refs[0]}` : 'greedy'
      }
    } catch {
      if (defaultCompetitive && !value) value = 'greedy'
    }
  })
</script>

<div class="polsel">
  <select value={selectValue} on:change={onSelect}>
    {#each groups as g}
      <optgroup label={g.label}>
        {#each g.options as o}<option value={o.value}>{o.label}</option>{/each}
      </optgroup>
    {/each}
    <option value={CUSTOM}>Custom spec…</option>
  </select>
  {#if custom || selectValue === CUSTOM}
    <PolicyInput bind:value placeholder="base:params, e.g. vbeam:depot:b0/b0_s0.zip,8,20" />
  {/if}
  {#if !compact && explanation}
    <div class="explain" class:bad={!explanation.known}>
      <p>{explanation.blurb}</p>
      {#if explanation.params.length}
        <ul>
          {#each explanation.params as p}
            <li>
              <code>{p.name}={p.value}</code>{#if p.isDefault}<span class="def"> (default)</span>{/if}
              — {p.meaning}
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  {/if}
</div>

<style>
  .polsel { display: flex; flex-direction: column; gap: 6px; }
  select { background: #23232b; color: #ddd; border: 1px solid #3a3f55; border-radius: 4px;
    padding: 6px 10px; font-size: 14px; min-width: 260px; }
  .explain { background: #14141a; border: 1px solid #23232b; border-radius: 6px;
    padding: 8px 12px; font-size: 12px; color: #9a9ab0; max-width: 460px; }
  .explain.bad { border-color: #6a3a4f; color: #e8a8a8; }
  .explain p { margin: 0; }
  .explain ul { margin: 6px 0 0; padding-left: 16px; }
  .explain li { margin: 2px 0; }
  code { color: #8fc7ff; font-family: ui-monospace, Consolas, monospace; }
  .def { color: #666; }
</style>
