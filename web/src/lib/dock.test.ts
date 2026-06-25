import { describe, expect, it } from 'vitest'
import { dockFalloff } from './dock'

describe('dockFalloff', () => {
  it('is maximal (1) at the cursor center', () => {
    expect(dockFalloff(0, 160)).toBeCloseTo(1)
  })
  it('is 0 at and beyond the radius', () => {
    expect(dockFalloff(160, 160)).toBeCloseTo(0)
    expect(dockFalloff(500, 160)).toBe(0)
  })
  it('decays monotonically with distance and is symmetric', () => {
    const a = dockFalloff(40, 160)
    const b = dockFalloff(80, 160)
    const c = dockFalloff(120, 160)
    expect(a).toBeGreaterThan(b)
    expect(b).toBeGreaterThan(c)
    expect(dockFalloff(-40, 160)).toBeCloseTo(a)
  })
})
