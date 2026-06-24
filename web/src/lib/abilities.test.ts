import { describe, expect, it } from 'vitest'
import { abilityList, hasAura } from './abilities'

describe('abilities', () => {
  it('lists present abilities in mask order with name + color', () => {
    const list = abilityList('B-D-L-')
    expect(list.map((a) => a.letter)).toEqual(['B', 'D', 'L'])
    expect(list[0]).toMatchObject({ name: 'Breakthrough', color: '#ff8a3d' })
  })

  it('returns empty for all-dashes or nullish input', () => {
    expect(abilityList('------')).toEqual([])
    expect(abilityList(undefined)).toEqual([])
    expect(abilityList(null)).toEqual([])
  })

  it('hasAura detects guard/ward/lethal and ignores absent', () => {
    expect(hasAura('---G--', 'G')).toBe(true)
    expect(hasAura('----L-', 'L')).toBe(true)
    expect(hasAura('-----W', 'W')).toBe(true)
    expect(hasAura('------', 'W')).toBe(false)
  })
})
