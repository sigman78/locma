<!-- Minimal dependency-free SVG line chart: multi-series, optional CI band,
     optional dot-series, zero line, hover readout. Series are named lists of
     [x, y] points (the shape the jobs API streams). -->
<script lang="ts">
  export let series: Record<string, [number, number][]> = {}
  export let band: [string, string] | null = null // [lo, hi] names, shaded not drawn
  export let dots: string[] = [] // series drawn as points instead of a line
  export let title = ''
  export let xLabel = ''
  export let height = 170
  export let width = 540

  const COLORS = ['#4fa3d9', '#4fd97a', '#d9a64f', '#d94f6b', '#9a6fd9', '#5fd9c9', '#c9d95f']
  const PAD = { l: 48, r: 12, t: 10, b: 22 }

  $: bandNames = band ? [band[0], band[1]] : []
  $: drawNames = Object.keys(series).filter((n) => !bandNames.includes(n) && series[n]?.length)
  $: all = Object.keys(series).flatMap((n) => series[n] ?? [])
  $: hasData = all.length > 0
  $: xmin = hasData ? Math.min(...all.map((p) => p[0])) : 0
  $: xmax = hasData ? Math.max(...all.map((p) => p[0])) : 1
  $: ymin0 = hasData ? Math.min(...all.map((p) => p[1])) : 0
  $: ymax0 = hasData ? Math.max(...all.map((p) => p[1])) : 1
  $: ypad = (ymax0 - ymin0 || Math.abs(ymax0) || 1) * 0.08
  $: ymin = ymin0 - ypad
  $: ymax = ymax0 + ypad
  $: innerW = width - PAD.l - PAD.r
  $: innerH = height - PAD.t - PAD.b
  $: sx = (x: number) => PAD.l + (xmax === xmin ? 0.5 : (x - xmin) / (xmax - xmin)) * innerW
  $: sy = (y: number) => PAD.t + (1 - (ymax === ymin ? 0.5 : (y - ymin) / (ymax - ymin))) * innerH

  const sorted = (n: string) => [...(series[n] ?? [])].sort((a, b) => a[0] - b[0])
  $: poly = (n: string) => sorted(n).map((p) => `${sx(p[0])},${sy(p[1])}`).join(' ')
  $: bandPath = band
    ? [...sorted(band[0]).map((p) => `${sx(p[0])},${sy(p[1])}`),
       ...sorted(band[1]).reverse().map((p) => `${sx(p[0])},${sy(p[1])}`)].join(' ')
    : ''
  $: yticks = [0, 1, 2, 3].map((i) => ymin + ((ymax - ymin) * i) / 3)
  $: xticks = [0, 1, 2].map((i) => xmin + ((xmax - xmin) * i) / 2)

  const fmt = (v: number) =>
    Math.abs(v) >= 1000 ? v.toFixed(0) : Number(v.toPrecision(3)).toString()

  let hover: { x: number; parts: string[] } | null = null
  function onMove(e: MouseEvent) {
    if (!hasData) return
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
    const fx = xmin + ((e.clientX - rect.left - PAD.l) / innerW) * (xmax - xmin)
    const parts = drawNames.map((n) => {
      const pts = sorted(n)
      let best = pts[0]
      for (const p of pts) if (Math.abs(p[0] - fx) < Math.abs(best[0] - fx)) best = p
      return `${n}=${fmt(best[1])}`
    })
    const nearest = sorted(drawNames[0] ?? '')[0]
    hover = { x: fx, parts: nearest ? [`x=${fmt(fx)}`, ...parts] : parts }
  }
</script>

<figure>
  <figcaption>
    <span>{title}</span>
    <span class="hover">{hover ? hover.parts.join('  ') : ''}</span>
  </figcaption>
  <svg
    {width}
    {height}
    role="img"
    aria-label={title}
    on:mousemove={onMove}
    on:mouseleave={() => (hover = null)}
  >
    {#if hasData}
      {#each yticks as t}
        <line x1={PAD.l} x2={width - PAD.r} y1={sy(t)} y2={sy(t)} class="grid" />
        <text x={PAD.l - 6} y={sy(t) + 3} class="tick" text-anchor="end">{fmt(t)}</text>
      {/each}
      {#each xticks as t}
        <text x={sx(t)} y={height - 6} class="tick" text-anchor="middle">{fmt(t)}</text>
      {/each}
      {#if ymin < 0 && ymax > 0}
        <line x1={PAD.l} x2={width - PAD.r} y1={sy(0)} y2={sy(0)} class="zero" />
      {/if}
      {#if band && series[band[0]]?.length}
        <polygon points={bandPath} class="band" />
      {/if}
      {#each drawNames as n, i}
        {#if dots.includes(n)}
          {#each series[n] ?? [] as p}
            <circle cx={sx(p[0])} cy={sy(p[1])} r="3" fill={COLORS[i % COLORS.length]} />
          {/each}
        {:else}
          <polyline points={poly(n)} fill="none" stroke={COLORS[i % COLORS.length]}
            stroke-width="1.6" />
        {/if}
      {/each}
    {:else}
      <text x={width / 2} y={height / 2} class="tick" text-anchor="middle">no data yet</text>
    {/if}
  </svg>
  <div class="legend">
    {#each drawNames as n, i}
      <span><i style="background: {COLORS[i % COLORS.length]}"></i>{n}</span>
    {/each}
    {#if xLabel}<span class="xlabel">x: {xLabel}</span>{/if}
  </div>
</figure>

<style>
  figure { margin: 0; }
  figcaption { display: flex; justify-content: space-between; font-size: 12px;
    color: #9a9ab0; margin-bottom: 2px; min-height: 16px; }
  .hover { color: #777; font-family: ui-monospace, Consolas, monospace; font-size: 11px; }
  svg { background: #101016; border: 1px solid #1e1e26; border-radius: 6px; display: block; }
  .grid { stroke: #1c1c24; }
  .zero { stroke: #444; stroke-dasharray: 4 3; }
  .tick { fill: #666; font-size: 10px; }
  .band { fill: rgba(79, 163, 217, 0.12); }
  .legend { display: flex; gap: 14px; font-size: 11px; color: #9a9ab0; margin-top: 3px;
    flex-wrap: wrap; }
  .legend i { display: inline-block; width: 10px; height: 3px; margin-right: 4px;
    vertical-align: middle; }
  .xlabel { color: #666; margin-left: auto; }
</style>
