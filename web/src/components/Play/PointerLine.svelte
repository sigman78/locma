<!-- web/src/components/Play/PointerLine.svelte -->
<script lang="ts">
  export let from: { x: number; y: number } | null = null
  export let to: { x: number; y: number } | null = null
</script>

{#if from && to}
  <svg class="pointer" width="100%" height="100%">
    <defs>
      <marker id="aim-head" markerWidth="10" markerHeight="10" refX="6" refY="3"
        orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L6,3 L0,6 Z" fill="#ffd23d" />
      </marker>
    </defs>
    <line x1={from.x} y1={from.y} x2={to.x} y2={to.y}
      stroke="#ffd23d" stroke-width="4" stroke-linecap="round"
      marker-end="url(#aim-head)" opacity="0.9" />
  </svg>
{/if}

<style>
  .pointer { position: fixed; inset: 0; z-index: 900; pointer-events: none; }
</style>
