import { describe, expect, it } from 'vitest'
import type { ActionDict, EventDict } from './replay'
import { computeFx } from './fx'
import {
  attackTargets,
  canSummon,
  cardDamage,
  faceDamage,
  itemTargets,
  lungeDirFor,
  splashesFor,
} from './play'

const legal: ActionDict[] = [
  { t: 'pass' },
  { t: 'summon', id: 5 },
  { t: 'attack', a: 7, target: -1 },
  { t: 'attack', a: 7, target: 12 },
  { t: 'use', item: 3, target: 9 },
]

describe('legal predicates', () => {
  it('canSummon', () => {
    expect(canSummon(legal, 5)).toBe(true)
    expect(canSummon(legal, 6)).toBe(false)
  })
  it('attackTargets includes face and units', () => {
    expect(attackTargets(legal, 7).sort((a, b) => a - b)).toEqual([-1, 12])
    expect(attackTargets(legal, 99)).toEqual([])
  })
  it('itemTargets', () => {
    expect(itemTargets(legal, 3)).toEqual([9])
  })
})

describe('splash mapping', () => {
  it('maps card damage by physical seat + iid', () => {
    const evs: EventDict[] = [{ t: 'damage', seat: 1, target: 12, amount: 4, fatal: false }]
    const sp = splashesFor(evs)
    expect(cardDamage(sp, 1, 12)).toBe(4)
    expect(cardDamage(sp, 0, 12)).toBe(null)
  })
  it('maps face damage by seat', () => {
    const evs: EventDict[] = [{ t: 'damage', seat: 0, target: 'face', amount: 3, fatal: false }]
    expect(faceDamage(splashesFor(evs), 0)).toBe(3)
    expect(faceDamage(splashesFor(evs), 1)).toBe(null)
  })
})

describe('lungeDirFor', () => {
  const attack: ActionDict = { t: 'attack', a: 7, target: -1 }
  it('my attacker lunges up, opponent attacker lunges down', () => {
    const mine = computeFx([], attack, 0) // seat 0 acts
    expect(lungeDirFor(mine, 0, 0, 7)).toBe('up') // you=0, card seat 0
    const theirs = computeFx([], attack, 1) // seat 1 acts
    expect(lungeDirFor(theirs, 0, 1, 7)).toBe('down') // you=0, card seat 1
  })
  it('returns null for a non-lunging card or no fx', () => {
    const mine = computeFx([], attack, 0)
    expect(lungeDirFor(mine, 0, 0, 99)).toBe(null)
    expect(lungeDirFor(null, 0, 0, 7)).toBe(null)
  })
})
