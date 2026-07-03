<!-- Win-rate matrix heatmap (row vs column); fills in live as league pairs finish. -->
<script lang="ts">
  export let names: string[] = []
  export let matrix: Record<string, Record<string, number>> = {}

  const cell = (a: string, b: string): number | null => {
    const v = matrix?.[a]?.[b]
    return typeof v === 'number' ? v : null
  }
  // 0 -> red, 0.5 -> neutral, 1 -> green, on the dark theme
  const bg = (v: number) => `hsl(${Math.round(120 * v)}, 42%, ${18 + 10 * Math.abs(v - 0.5)}%)`
</script>

<table>
  <thead>
    <tr><th></th>{#each names as n}<th title={n}>{n}</th>{/each}</tr>
  </thead>
  <tbody>
    {#each names as row}
      <tr>
        <th title={row}>{row}</th>
        {#each names as col}
          {@const v = row === col ? null : cell(row, col)}
          {#if row === col}
            <td class="diag">—</td>
          {:else if v === null}
            <td class="empty">·</td>
          {:else}
            <td style="background: {bg(v)}">{v.toFixed(2)}</td>
          {/if}
        {/each}
      </tr>
    {/each}
  </tbody>
</table>

<style>
  table { border-collapse: collapse; font-size: 12px; margin-top: 6px; }
  th, td { padding: 4px 8px; text-align: center; border: 1px solid #14141a; }
  th { color: #777; font-weight: 500; font-family: ui-monospace, Consolas, monospace;
    max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    text-align: left; }
  thead th { text-align: center; }
  td { color: #ddd; font-family: ui-monospace, Consolas, monospace; }
  .diag { color: #555; background: #101016; }
  .empty { color: #444; background: #101016; }
</style>
