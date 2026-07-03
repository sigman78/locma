<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import PolicySelect from '../shared/PolicySelect.svelte'

  const dispatch = createEventDispatcher<{ start: { opponent: string; seed?: number } }>()
  let opponent = ''
  let seedStr = ''

  function go() {
    const seed = seedStr.trim() === '' ? undefined : Number(seedStr)
    dispatch('start', { opponent, seed })
  }
</script>

<div class="newgame">
  <div class="row">
    <label class="opp">
      Opponent
      <PolicySelect bind:value={opponent} defaultCompetitive />
    </label>
    <label>
      Seed
      <input bind:value={seedStr} placeholder="random" />
    </label>
    <button on:click={go} disabled={!opponent.trim()}>Start game</button>
  </div>
</div>

<style>
  .newgame { background: #15151b; border: 1px solid #2a2a36; border-radius: 8px;
    padding: 16px; }
  .row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
  label { display: flex; flex-direction: column; gap: 4px; font-size: 14px; color: #bbb; }
  input { background: #23232b; color: #ddd; border: 1px solid #3a3f55;
    border-radius: 4px; padding: 6px 10px; font-size: 15px; width: 90px; }
  button { background: #2a2a44; color: #fff; border: 1px solid #4a4f6a; border-radius: 4px;
    padding: 7px 16px; cursor: pointer; font-weight: 600; margin-top: 22px; }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
