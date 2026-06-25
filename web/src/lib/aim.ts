export interface AimTarget {
  id: number | 'face'
  x: number
  y: number
}

/** The legal target nearest the cursor within `snapRadius`, else null. */
export function nearestTarget(
  cursorX: number,
  cursorY: number,
  targets: AimTarget[],
  snapRadius = 70,
): AimTarget | null {
  let best: AimTarget | null = null
  let bestD = snapRadius
  for (const t of targets) {
    const d = Math.hypot(t.x - cursorX, t.y - cursorY)
    if (d <= bestD) {
      bestD = d
      best = t
    }
  }
  return best
}
