<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte'
  import { getPolicies } from '../../lib/api'

  const dispatch = createEventDispatcher<{ start: { opponent: string; seed?: number } }>()
  let policies: string[] = []
  let opponent = ''
  let seedStr = ''

  onMount(async () => {
    policies = await getPolicies()
    opponent = policies[0] ?? ''
  })

  function go() {
    const seed = seedStr.trim() === '' ? undefined : Number(seedStr)
    dispatch('start', { opponent, seed })
  }
</script>

<div class="newgame">
  <label>
    Opponent
    <select bind:value={opponent}>
      {#each policies as p}<option value={p}>{p}</option>{/each}
    </select>
  </label>
  <label>
    Seed
    <input bind:value={seedStr} placeholder="random" />
  </label>
  <button on:click={go} disabled={!opponent}>Start game</button>
</div>

<style>
  .newgame { display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap;
    background: #15151b; border: 1px solid #2a2a36; border-radius: 8px; padding: 16px; }
  label { display: flex; flex-direction: column; gap: 4px; font-size: 14px; color: #bbb; }
  select, input { background: #23232b; color: #ddd; border: 1px solid #3a3f55;
    border-radius: 4px; padding: 6px 10px; font-size: 15px; }
  button { background: #2a2a44; color: #fff; border: 1px solid #4a4f6a; border-radius: 4px;
    padding: 7px 16px; cursor: pointer; font-weight: 600; }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
