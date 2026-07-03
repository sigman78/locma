<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import PolicyInput from '../shared/PolicyInput.svelte'

  const dispatch = createEventDispatcher<{ start: { opponent: string; seed?: number } }>()
  let opponent = 'greedy'
  let seedStr = ''

  function go() {
    const seed = seedStr.trim() === '' ? undefined : Number(seedStr)
    dispatch('start', { opponent, seed })
  }
</script>

<div class="newgame">
  <label>
    Opponent
    <PolicyInput bind:value={opponent} placeholder="greedy, vbeam:depot:b0/b0_s0.zip, ..." />
  </label>
  <label>
    Seed
    <input bind:value={seedStr} placeholder="random" />
  </label>
  <button on:click={go} disabled={!opponent.trim()}>Start game</button>
</div>

<style>
  .newgame { display: flex; gap: 16px; align-items: flex-end; flex-wrap: wrap;
    background: #15151b; border: 1px solid #2a2a36; border-radius: 8px; padding: 16px; }
  label { display: flex; flex-direction: column; gap: 4px; font-size: 14px; color: #bbb; }
  input { background: #23232b; color: #ddd; border: 1px solid #3a3f55;
    border-radius: 4px; padding: 6px 10px; font-size: 15px; width: 90px; }
  button { background: #2a2a44; color: #fff; border: 1px solid #4a4f6a; border-radius: 4px;
    padding: 7px 16px; cursor: pointer; font-weight: 600; }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
