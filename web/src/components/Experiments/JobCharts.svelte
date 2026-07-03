<!-- Live visualization for a job, chosen by kind: convergence curves for
     matches, paired deltas for ceiling verdicts, a filling heatmap for
     leagues, small-multiple training curves for train-zoo. -->
<script lang="ts">
  import Chart from '../shared/Chart.svelte'
  import Heatmap from '../shared/Heatmap.svelte'

  export let kind: string
  export let series: Record<string, [number, number][]> = {}
  export let live: Record<string, any> = {}

  const TRAIN_HIDE = new Set(['timesteps'])
  $: trainNames = Object.keys(series).filter((n) => !TRAIN_HIDE.has(n) && series[n]?.length)

  const pick = (names: string[]) =>
    Object.fromEntries(names.filter((n) => series[n]).map((n) => [n, series[n]]))
</script>

{#if kind === 'match' || kind === 'noise-floor'}
  <Chart
    title="win rate convergence"
    series={pick(['win_rate', 'ci_lo', 'ci_hi'])}
    band={['ci_lo', 'ci_hi']}
    xLabel="games played"
  />
{:else if kind === 'ceiling'}
  <Chart
    title="paired delta per seed (candidate − baseline)"
    series={pick(['delta', 'mean_delta'])}
    dots={['delta']}
    xLabel="seed index / seeds done"
  />
{:else if kind === 'league'}
  {#if live?.matrix}
    <Heatmap names={live.policies ?? []} matrix={live.matrix} />
  {/if}
{:else if kind === 'train-zoo'}
  <div class="grid">
    {#each trainNames as n}
      <Chart title={n} series={{ [n]: series[n] }} width={340} height={140}
        xLabel="timesteps" />
    {/each}
  </div>
{/if}

<style>
  .grid { display: flex; flex-wrap: wrap; gap: 14px; }
</style>
