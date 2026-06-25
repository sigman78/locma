/** macOS-Dock magnification curve: 1 at the cursor, smoothly → 0 at `radius`. */
export function dockFalloff(distance: number, radius: number): number {
  const d = Math.abs(distance)
  if (d >= radius) return 0
  return (Math.cos((d / radius) * Math.PI) + 1) / 2
}

export interface DockOpts {
  amp?: number
  radius?: number
  enabled?: boolean
}

/** Svelte action: magnify the hovered child (and taper its neighbours). */
export function dock(node: HTMLElement, opts: DockOpts = {}) {
  let amp = opts.amp ?? 0.18
  let radius = opts.radius ?? 160
  let enabled = opts.enabled ?? true

  const kids = () => Array.from(node.children) as HTMLElement[]

  function reset() {
    for (const c of kids()) {
      c.style.transform = ''
      c.style.zIndex = ''
    }
  }
  function onMove(e: MouseEvent) {
    if (!enabled) return
    for (const c of kids()) {
      const r = c.getBoundingClientRect()
      const f = dockFalloff(e.clientX - (r.left + r.width / 2), radius)
      c.style.transform = `scale(${1 + amp * f}) translateY(${-amp * 22 * f}px)`
      c.style.zIndex = f > 0.5 ? '20' : ''
    }
  }
  node.addEventListener('mousemove', onMove)
  node.addEventListener('mouseleave', reset)

  return {
    update(next: DockOpts) {
      amp = next.amp ?? amp
      radius = next.radius ?? radius
      enabled = next.enabled ?? true
      if (!enabled) reset()
    },
    destroy() {
      node.removeEventListener('mousemove', onMove)
      node.removeEventListener('mouseleave', reset)
    },
  }
}
