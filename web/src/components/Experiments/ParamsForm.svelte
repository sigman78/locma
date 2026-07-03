<!-- Schema-driven experiment parameter form: the server's kind schema decides
     which fields exist, so new experiment kinds need no frontend changes. -->
<script lang="ts">
  import type { SchemaField } from '../../lib/api'
  import PolicyInput from '../shared/PolicyInput.svelte'
  import PolicySelect from '../shared/PolicySelect.svelte'

  export let schema: SchemaField[] = []
  export let params: Record<string, any> = {}

  function addItem(name: string) {
    params[name] = [...(params[name] ?? []), '']
  }
  function removeItem(name: string, i: number) {
    params[name] = (params[name] as string[]).filter((_, k) => k !== i)
  }
</script>

<div class="form">
  {#each schema as f}
    <label class="field" class:wide={f.type === 'policies'}>
      <span class="name" title={f.help ?? ''}>{f.name}{#if f.help}<em> — {f.help}</em>{/if}</span>
      {#if f.type === 'policy'}
        <PolicySelect bind:value={params[f.name]} compact />
      {:else if f.type === 'policies'}
        <div class="list">
          {#each params[f.name] ?? [] as _, i}
            <div class="row">
              <PolicyInput bind:value={params[f.name][i]} />
              <button type="button" class="mini" on:click={() => removeItem(f.name, i)}>x</button>
            </div>
          {/each}
          <button type="button" class="mini add" on:click={() => addItem(f.name)}>+ add</button>
        </div>
      {:else if f.type === 'int' || f.type === 'float'}
        <input type="number" step={f.type === 'float' ? 'any' : '1'} bind:value={params[f.name]} />
      {:else}
        <input bind:value={params[f.name]} />
      {/if}
    </label>
  {/each}
</div>

<style>
  .form { display: flex; flex-wrap: wrap; gap: 12px 20px; }
  .field { display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: #bbb; }
  .field.wide { flex-basis: 100%; }
  .name { color: #9a9ab0; }
  .name em { color: #666; font-style: normal; }
  input { background: #23232b; color: #ddd; border: 1px solid #3a3f55; border-radius: 4px;
    padding: 6px 10px; font-size: 13px; width: 110px; }
  .list { display: flex; flex-direction: column; gap: 6px; align-items: flex-start; }
  .row { display: flex; gap: 6px; align-items: center; }
  .mini { background: #23232b; color: #999; border: 1px solid #333; border-radius: 4px;
    padding: 4px 8px; cursor: pointer; font-size: 12px; }
  .mini:hover { color: #ddd; }
  .add { margin-top: 2px; }
</style>
