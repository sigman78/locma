<!-- Unified policy combobox: one input that is both dropdown and editor.
     Focus lists curated presets (depot models, baselines, search); typing
     filters them; a grayed inline ghost completes the base and shows the
     remaining argument syntax; once arguments are being typed, a floating
     panel underneath explains the base and highlights the parameter under
     the caret. Tab accepts the ghost completion. -->
<script lang="ts">
  import { onMount } from 'svelte'
  import { policyCatalog } from '../../lib/catalog'
  import { SPEC_INFO } from '../../lib/specs'

  export let value = ''
  export let placeholder = 'select or type a policy'
  /** Pick the strongest depot-backed spec as the initial value (Play mode). */
  export let defaultCompetitive = false

  interface Option {
    value: string
    label: string
    group: string
  }

  const STATIC: Option[] = [
    { value: 'greedy', label: 'Greedy', group: 'baseline' },
    { value: 'scripted', label: 'Scripted', group: 'baseline' },
    { value: 'max-guard', label: 'Max-Guard', group: 'baseline' },
    { value: 'max-attack', label: 'Max-Attack', group: 'baseline' },
    { value: 'random', label: 'Random', group: 'baseline' },
    { value: 'dmcts:15,30', label: 'DMCTS (fair search)', group: 'search' },
    { value: 'mcts:100', label: 'MCTS (cheating search)', group: 'search' },
    { value: 'azlite:100', label: 'AZ-lite (PUCT)', group: 'search' },
  ]
  let options: Option[] = STATIC
  let open = false
  let hi = 0
  let inputEl: HTMLInputElement

  onMount(async () => {
    try {
      const cat = await policyCatalog()
      const models: Option[] = []
      for (const m of cat.depot_models) {
        const ref = m.refs[0]
        if (!ref) continue
        models.push({ value: `vbeam:${ref}`, label: `V-beam · ${m.name} (strongest)`, group: 'model' })
        models.push({ value: `ppo:${ref}`, label: `PPO · ${m.name} (reactive)`, group: 'model' })
      }
      options = [...models, ...STATIC]
      if (defaultCompetitive && !value) {
        const b0 = cat.depot_models.find((m) => m.name === 'b0') ?? cat.depot_models[0]
        value = b0?.refs[0] ? `vbeam:${b0.refs[0]}` : 'greedy'
      }
    } catch {
      if (defaultCompetitive && !value) value = 'greedy'
    }
  })

  // -- parsing the current text ------------------------------------------------
  $: ci = value.indexOf(':')
  $: base = ci === -1 ? value.trim() : value.slice(0, ci)
  $: inArgs = ci !== -1
  $: baseInfo = SPEC_INFO[base]
  $: curIdx = inArgs ? value.slice(ci + 1).split(',').length - 1 : -1

  // -- dropdown filtering (base-typing mode only) -------------------------------
  $: q = value.trim().toLowerCase()
  $: filtered = inArgs
    ? []
    : options.filter(
        (o) =>
          !q ||
          o.value.toLowerCase().includes(q) ||
          o.label.toLowerCase().includes(q),
      )
  $: if (hi >= filtered.length) hi = Math.max(0, filtered.length - 1)

  // -- inline ghost: base completion + remaining argument syntax ---------------
  const argSyntax = (b: string, from: number) =>
    (SPEC_INFO[b]?.params ?? []).slice(from).map((p) => p.name).join(',')

  $: ghost = (() => {
    if (!value.trim()) return ''
    if (!inArgs) {
      const completion = Object.keys(SPEC_INFO).find((k) => k.startsWith(base) && k !== base)
      const target = completion ?? (SPEC_INFO[base] ? base : null)
      if (target === null) return ''
      const syntax = argSyntax(target, 0)
      return target.slice(base.length) + (syntax ? `:${syntax}` : '')
    }
    if (!baseInfo) return ''
    const parts = value.slice(ci + 1).split(',')
    const rest = argSyntax(base, parts.length)
    if (parts[parts.length - 1] === '') {
      const cur = baseInfo.params[parts.length - 1]?.name ?? ''
      return [cur, rest].filter(Boolean).join(',')
    }
    return rest ? `,${rest}` : ''
  })()

  // -- interactions --------------------------------------------------------------
  function pick(o: Option) {
    value = o.value
    open = false
  }
  function acceptGhost() {
    if (!ghost || inArgs) return false
    const completion = Object.keys(SPEC_INFO).find((k) => k.startsWith(base) && k !== base)
    const target = completion ?? base
    if (!SPEC_INFO[target]) return false
    value = target + (SPEC_INFO[target].params.length ? ':' : '')
    return true
  }
  function onKey(e: KeyboardEvent) {
    if (e.key === 'Tab' && !e.shiftKey && ghost) {
      if (acceptGhost()) e.preventDefault()
    } else if (e.key === 'ArrowDown' && filtered.length) {
      open = true
      hi = Math.min(hi + 1, filtered.length - 1)
      e.preventDefault()
    } else if (e.key === 'ArrowUp' && filtered.length) {
      hi = Math.max(hi - 1, 0)
      e.preventDefault()
    } else if (e.key === 'Enter') {
      if (open && filtered[hi] && !inArgs) {
        pick(filtered[hi])
        e.preventDefault()
      } else {
        open = false
      }
    } else if (e.key === 'Escape') {
      open = false
    } else {
      open = true
    }
  }
</script>

<div class="combo">
  <div class="inputwrap">
    <div class="ghostline" aria-hidden="true">
      <span class="typed">{value}</span><span class="suffix">{ghost}</span>
    </div>
    <input
      bind:this={inputEl}
      bind:value
      {placeholder}
      spellcheck="false"
      autocomplete="off"
      on:focus={() => (open = true)}
      on:keydown={onKey}
      on:blur={() => (open = false)}
    />
    <button
      type="button"
      class="chev"
      tabindex="-1"
      aria-label="show policies"
      on:mousedown|preventDefault={() => {
        open = !open
        inputEl.focus()
      }}
    >
      <svg width="10" height="6" viewBox="0 0 10 6"><path d="M1 1l4 4 4-4" fill="none"
        stroke="currentColor" stroke-width="1.5" /></svg>
    </button>
  </div>

  {#if open}
    <div class="panel">
      {#if !inArgs && filtered.length}
        {#each filtered as o, i}
          <button
            type="button"
            class="row"
            class:hi={i === hi}
            on:mousedown|preventDefault={() => pick(o)}
            on:mousemove={() => (hi = i)}
          >
            <span class="olabel">{o.label}</span>
            <span class="ovalue">{o.value}</span>
            <span class="ogroup">{o.group}</span>
          </button>
        {/each}
      {:else if baseInfo}
        <div class="hint">
          <p class="blurb">{baseInfo.blurb}</p>
          {#if baseInfo.params.length}
            <ul>
              {#each baseInfo.params as p, i}
                <li class:cur={i === curIdx}>
                  <code>{p.name}</code>
                  <span class="def">(default {p.default})</span> — {p.meaning}
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      {:else if value.trim()}
        <div class="hint bad"><p class="blurb">Unknown policy base '{base}' — check the spec.</p></div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .combo { position: relative; display: inline-block; min-width: 320px; }
  .inputwrap { position: relative; }
  /* the ghost mirrors the input's font/padding; the typed part is invisible so
     the gray suffix starts exactly where the caret is */
  .ghostline, input {
    font-family: ui-monospace, Consolas, monospace; font-size: 13px;
    padding: 7px 26px 7px 10px; box-sizing: border-box; width: 100%;
    white-space: pre; overflow: hidden;
  }
  .ghostline { position: absolute; inset: 0; pointer-events: none;
    border: 1px solid transparent; border-radius: 4px; }
  .typed { color: transparent; }
  .suffix { color: #565666; }
  input { position: relative; background: #23232b; color: #ddd;
    border: 1px solid #3a3f55; border-radius: 4px; }
  input:focus { outline: none; border-color: #4a4f8a; }
  /* input paints over the ghost unless its own background is see-through */
  .inputwrap input { background: transparent; }
  .inputwrap { background: #23232b; border-radius: 4px; }
  .chev { position: absolute; right: 2px; top: 0; bottom: 0; width: 22px;
    background: none; border: none; color: #666; cursor: pointer; }
  .chev:hover { color: #bbb; }

  .panel { position: absolute; top: calc(100% + 4px); left: 0; min-width: 100%;
    z-index: 120; background: #16161d; border: 1px solid #2e2e3a; border-radius: 6px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.55); max-height: 300px; overflow-y: auto; }
  .row { display: flex; gap: 10px; align-items: baseline; width: 100%;
    background: none; border: none; padding: 6px 10px; cursor: pointer;
    text-align: left; font-size: 13px; color: #ccc; }
  .row.hi { background: #23233a; }
  .olabel { flex-shrink: 0; }
  .ovalue { color: #667; font-family: ui-monospace, Consolas, monospace; font-size: 11px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .ogroup { color: #555; font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.06em; flex-shrink: 0; }

  .hint { padding: 8px 12px; font-size: 12px; color: #9a9ab0; max-width: 460px; }
  .hint.bad { color: #e8a8a8; }
  .blurb { margin: 0; }
  .hint ul { margin: 6px 0 0; padding-left: 16px; }
  .hint li { margin: 2px 0; }
  .hint li.cur { color: #ddd; background: #20203a; border-radius: 3px;
    padding: 1px 4px; margin-left: -4px; list-style-position: inside; }
  code { color: #8fc7ff; font-family: ui-monospace, Consolas, monospace; }
  .def { color: #666; }
</style>
