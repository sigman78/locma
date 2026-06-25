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
  /** CSS selector for the element to scale within each child (default: the child itself).
   *  Use this to scale the card visual while leaving its tooltip un-scaled. */
  target?: string
}

/** Svelte action: magnify the hovered child (and taper its neighbours). */
export function dock(node: HTMLElement, opts: DockOpts = {}) {
  let amp = opts.amp ?? 0.18
  let radius = opts.radius ?? 160
  let enabled = opts.enabled ?? true
  let target = opts.target

  const kids = () => Array.from(node.children) as HTMLElement[]
  // the element to transform within a child (the card), and the child itself for z-index
  const visual = (c: HTMLElement): HTMLElement => (target ? (c.querySelector<HTMLElement>(target) ?? c) : c)

  function reset() {
    for (const c of kids()) {
      visual(c).style.transform = ''
      c.style.zIndex = ''
    }
  }
  function onMove(e: MouseEvent) {
    if (!enabled) return
    for (const c of kids()) {
      const r = c.getBoundingClientRect()
      const f = dockFalloff(e.clientX - (r.left + r.width / 2), radius)
      visual(c).style.transform = `scale(${1 + amp * f}) translateY(${-amp * 22 * f}px)`
      c.style.zIndex = f > 0.5 ? '20' : ''
    }
  }
  node.addEventListener('mousemove', onMove)
  node.addEventListener('mouseleave', reset)

  return {
    update(next: DockOpts) {
      amp = next.amp ?? amp
      radius = next.radius ?? radius
      enabled = next.enabled ?? enabled
      target = next.target ?? target
      if (!enabled) reset()
    },
    destroy() {
      node.removeEventListener('mousemove', onMove)
      node.removeEventListener('mouseleave', reset)
    },
  }
}
