import { describe, expect, it } from 'vitest'
import { nearestTarget, type AimTarget } from './aim'

const targets: AimTarget[] = [
  { id: 12, x: 100, y: 100 },
  { id: 'face', x: 300, y: 50 },
]

describe('nearestTarget', () => {
  it('snaps to a target within the radius', () => {
    expect(nearestTarget(110, 105, targets, 70)?.id).toBe(12)
  })
  it('returns null when the cursor is outside every target radius', () => {
    expect(nearestTarget(700, 700, targets, 70)).toBe(null)
  })
  it('prefers the closer of two in-range targets', () => {
    const near: AimTarget[] = [
      { id: 1, x: 0, y: 0 },
      { id: 2, x: 20, y: 0 },
    ]
    expect(nearestTarget(18, 0, near, 70)?.id).toBe(2)
  })
})
