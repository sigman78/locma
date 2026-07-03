<!-- Free-form policy spec input with catalog suggestions (baselines, search
     presets, depot-backed model specs like vbeam:depot:b0/b0_s0.zip). -->
<script lang="ts">
  import { onMount } from 'svelte'
  import { policyCatalog } from '../../lib/catalog'

  export let value = ''
  export let placeholder = 'policy spec'

  let suggestions: string[] = []
  const listId = `policies-${Math.random().toString(36).slice(2, 9)}`

  onMount(async () => {
    try {
      suggestions = (await policyCatalog()).suggestions
    } catch {
      suggestions = []
    }
  })
</script>

<input list={listId} bind:value {placeholder} spellcheck="false" />
<datalist id={listId}>
  {#each suggestions as s}<option value={s}></option>{/each}
</datalist>

<style>
  input { background: #23232b; color: #ddd; border: 1px solid #3a3f55; border-radius: 4px;
    padding: 6px 10px; font-size: 13px; font-family: ui-monospace, Consolas, monospace;
    min-width: 240px; }
</style>
