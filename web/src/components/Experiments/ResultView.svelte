<!-- Per-kind experiment result rendering; unknown kinds fall back to JSON. -->
<script lang="ts">
  export let kind: string
  export let result: Record<string, any>

  const pct = (v: number) => (v ?? 0).toFixed(3)

  $: matrixNames = (result?.policies ?? []) as string[]
</script>

{#if kind === 'match' || kind === 'noise-floor'}
  <div class="card">
    <div class="big">
      win rate <strong>{pct(result.win_rate)}</strong>
      <span class="dim">95% CI [{pct(result.ci_lo)}, {pct(result.ci_hi)}]</span>
    </div>
    <div class="dim">
      {result.policy_a} vs {result.policy_b} — {result.wins_a}/{result.games} games,
      p={Number(result.p_value).toPrecision(3)}
      {#if kind === 'noise-floor'}
        · resolution limit ±{pct(result.resolution)}
      {/if}
    </div>
  </div>
{:else if kind === 'league'}
  <table>
    <thead>
      <tr><th>policy</th><th>openskill</th><th>elo</th><th>avg wr</th><th>p vs ref</th></tr>
    </thead>
    <tbody>
      {#each result.table as row}
        <tr>
          <td class="mono">{row.policy}</td>
          <td>{Number(row.openskill).toFixed(2)}</td>
          <td>{Number(row.elo).toFixed(0)}</td>
          <td>{pct(row.avg_win_rate)}</td>
          <td>{row.p_vs_ref == null ? '—' : Number(row.p_vs_ref).toPrecision(3)}</td>
        </tr>
      {/each}
    </tbody>
  </table>
  <table class="matrix">
    <thead>
      <tr><th></th>{#each matrixNames as n}<th class="mono">{n}</th>{/each}</tr>
    </thead>
    <tbody>
      {#each matrixNames as row}
        <tr>
          <td class="mono">{row}</td>
          {#each matrixNames as col}
            <td>{row === col ? '—' : pct(result.matrix[row][col])}</td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
{:else if kind === 'ceiling'}
  <div class="card">
    <div class="big">
      <strong class:good={result.verdict === 'headroom'}>VERDICT: {result.verdict}</strong>
    </div>
    <div class="dim">
      cand={pct(result.cand_avg)} base={pct(result.b0_avg)}
      delta={result.mean_delta >= 0 ? '+' : ''}{pct(result.mean_delta)}
      95% CI [{result.ci_lo >= 0 ? '+' : ''}{pct(result.ci_lo)},
      {result.ci_hi >= 0 ? '+' : ''}{pct(result.ci_hi)}]
    </div>
    <div class="dim mono">
      {result.candidates.join(', ')} vs {result.baselines.join(', ')}
      (opponents: {result.opponents.join(', ')})
    </div>
  </div>
{:else}
  <pre>{JSON.stringify(result, null, 2)}</pre>
{/if}

<style>
  .card { display: flex; flex-direction: column; gap: 4px; }
  .big { font-size: 15px; }
  .big strong { color: #eee; }
  .big strong.good { color: #4fd97a; }
  .dim { color: #888; font-size: 13px; }
  .mono { font-family: ui-monospace, Consolas, monospace; }
  table { border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th, td { text-align: left; padding: 3px 12px 3px 0; border-bottom: 1px solid #222; }
  th { color: #777; font-weight: 500; }
  .matrix td, .matrix th { padding-right: 14px; }
  pre { font-size: 12px; color: #aaa; overflow-x: auto; }
</style>
