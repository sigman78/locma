import { describe, expect, it, vi } from 'vitest'
import { playFrames } from './playback'

describe('playFrames', () => {
  it('calls onFrame for each frame in order, then onDone', () => {
    vi.useFakeTimers()
    const seen: number[] = []
    let done = false
    playFrames([10, 20, 30], (f) => seen.push(f), { holdMs: 100, onDone: () => (done = true) })
    expect(seen).toEqual([10]) // first frame fires synchronously
    vi.advanceTimersByTime(100)
    expect(seen).toEqual([10, 20])
    vi.advanceTimersByTime(100)
    expect(seen).toEqual([10, 20, 30])
    expect(done).toBe(false)
    vi.advanceTimersByTime(100)
    expect(done).toBe(true)
    vi.useRealTimers()
  })

  it('cancel() stops further frames and onDone', () => {
    vi.useFakeTimers()
    const seen: number[] = []
    let done = false
    const s = playFrames([1, 2, 3], (f) => seen.push(f), { holdMs: 50, onDone: () => (done = true) })
    expect(seen).toEqual([1])
    s.cancel()
    vi.advanceTimersByTime(500)
    expect(seen).toEqual([1])
    expect(done).toBe(false)
    vi.useRealTimers()
  })

  it('empty frame list calls onDone immediately', () => {
    vi.useFakeTimers()
    let done = false
    playFrames<number>([], () => {}, { onDone: () => (done = true) })
    expect(done).toBe(true)
    vi.useRealTimers()
  })
})
