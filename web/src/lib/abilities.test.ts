import { describe, expect, it } from 'vitest'
import { abilityList, hasAura, auraSplit } from './abilities'

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

describe('auraSplit', () => {
  it('G+L+D mask → taunt+lethal+pills=[Drain]', () => {
    // BCDGLW order: B=-,C=-,D=D,G=G,L=L,W=-
    const r = auraSplit('--DGL-')
    expect(r.taunt).toBe(true)
    expect(r.lethal).toBe(true)
    expect(r.ward).toBe(false)
    expect(r.pills.map((a) => a.letter)).toEqual(['D'])
  })

  it('all-dashes → all false, pills=[]', () => {
    const r = auraSplit('------')
    expect(r).toEqual({ taunt: false, ward: false, lethal: false, pills: [] })
  })

  it('null → all false, pills=[]', () => {
    const r = auraSplit(null)
    expect(r).toEqual({ taunt: false, ward: false, lethal: false, pills: [] })
  })

  it('W-only mask → ward true, others false, pills=[]', () => {
    const r = auraSplit('-----W')
    expect(r.ward).toBe(true)
    expect(r.taunt).toBe(false)
    expect(r.lethal).toBe(false)
    expect(r.pills).toEqual([])
  })

  it('B+C mask → both pills present, no auras', () => {
    const r = auraSplit('BC----')
    expect(r.taunt).toBe(false)
    expect(r.ward).toBe(false)
    expect(r.lethal).toBe(false)
    expect(r.pills.map((a) => a.letter)).toEqual(['B', 'C'])
  })
})
