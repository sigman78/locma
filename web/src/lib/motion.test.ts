import { describe, expect, it } from 'vitest'
import { animate, dealIn, spring } from './motion'

const node = {} as HTMLElement // these transitions ignore the node (approximated, no FLIP)

describe('spring / dealIn enter transitions', () => {
  it('are inert when the animate window is closed', () => {
    animate.set(false)
    expect(spring(node).duration).toBe(0)
    expect(dealIn(node).duration).toBe(0)
  })
  it('spring lowers the enlarged minion into its slot and settles at full size', () => {
    animate.set(true)
    const r = spring(node)
    expect(r.duration).toBeGreaterThan(0)
    expect(r.css!(0)).toContain('scale(1.25')
    expect(r.css!(0)).toContain('translateY(-34px)')
    expect(r.css!(1)).toContain('scale(1)')
    // backOut overshoot (t > 1) must never emit a negative drop-shadow blur
    expect(r.css!(1.1)).not.toMatch(/drop-shadow\(0 -?\d[\d.]*px -/)
    animate.set(false)
  })
  it('dealIn slides+flips in from the right', () => {
    animate.set(true)
    const r = dealIn(node)
    expect(r.duration).toBeGreaterThan(0)
    expect(r.css!(0)).toMatch(/rotateY/)
    expect(r.css!(0)).toMatch(/translateX/)
    expect(r.css!(1)).toMatch(/translateX\(0px\)/)
    animate.set(false)
  })
})
