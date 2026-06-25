export interface Sequencer {
  cancel: () => void
}

/**
 * Play `frames` one at a time: the first synchronously, each next one after
 * `holdMs`. `onDone` fires after the final frame's hold. `cancel()` halts
 * immediately. Pure scheduler — the caller's `onFrame` drives the animation
 * window (pulse) so this stays DOM-free and testable.
 */
export function playFrames<T>(
  frames: T[],
  onFrame: (frame: T, index: number) => void,
  opts: { holdMs?: number; onDone?: () => void } = {},
): Sequencer {
  const hold = opts.holdMs ?? 650
  let i = 0
  let timer: ReturnType<typeof setTimeout> | null = null
  let cancelled = false

  function tick() {
    if (cancelled) return
    if (i >= frames.length) {
      opts.onDone?.()
      return
    }
    onFrame(frames[i], i)
    i += 1
    timer = setTimeout(tick, hold)
  }
  tick()

  return {
    cancel() {
      cancelled = true
      if (timer) {
        clearTimeout(timer)
        timer = null
      }
    },
  }
}
