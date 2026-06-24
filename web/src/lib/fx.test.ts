import { describe, expect, it } from 'vitest'
import { computeFx } from './fx'
import type { Snapshot } from './replay'

const card = (iid: number, def: number): any => ({
  iid, card_id: iid, atk: 1, def, abilities: '------',
})
const player = (health: number, board: any[]): any => ({
  health, mana: 0, max_mana: 0, next_rune: 0, bonus_draw: 0, deck_count: 0, hand: [], board,
})
const snap = (p0: any, p1: any): Snapshot => ({ current: 0, players: [p0, p1] })

describe('computeFx', () => {
  it('numeric splash when a minion survives with less def', () => {
    const prev = snap(player(30, []), player(30, [card(7, 5)]))
    const next = snap(player(30, []), player(30, [card(7, 2)]))
    const fx = computeFx(prev, next, { t: 'attack', a: 0, target: 0 }, 0)
    expect(fx.splashes).toContainEqual({ seat: 1, target: 7, amount: 3, fatal: false })
  })

  it('fatal splash when a minion is removed', () => {
    const prev = snap(player(30, []), player(30, [card(7, 5)]))
    const next = snap(player(30, []), player(30, []))
    const fx = computeFx(prev, next, { t: 'attack', a: 0, target: 0 }, 0)
    expect(fx.splashes).toContainEqual({ seat: 1, target: 7, amount: 0, fatal: true })
  })

  it('face splash on health loss', () => {
    const prev = snap(player(30, []), player(30, []))
    const next = snap(player(30, []), player(26, []))
    const fx = computeFx(prev, next, { t: 'attack', a: 0, target: -1 }, 0)
    expect(fx.splashes).toContainEqual({ seat: 1, target: 'face', amount: 4, fatal: false })
  })

  it('lunge toward an enemy minion for attacks (resolved to attacker iid)', () => {
    const prev = snap(player(30, [card(1, 3)]), player(30, [card(7, 5)]))
    const next = snap(player(30, [card(1, 3)]), player(30, [card(7, 2)]))
    const fx = computeFx(prev, next, { t: 'attack', a: 1, target: 7 }, 0)
    expect(fx.lunge).toEqual({ seat: 0, iid: 1, toward: { seat: 1, iid: 7 } })
  })

  it('lunge toward face when target is -1', () => {
    const prev = snap(player(30, [card(1, 3)]), player(30, []))
    const next = snap(player(30, [card(1, 3)]), player(26, []))
    const fx = computeFx(prev, next, { t: 'attack', a: 1, target: -1 }, 0)
    expect(fx.lunge).toEqual({ seat: 0, iid: 1, toward: 'face' })
  })

  it('cast for use actions; no lunge', () => {
    const prev = snap(player(30, []), player(30, []))
    const next = snap(player(30, []), player(30, []))
    const fx = computeFx(prev, next, { t: 'use', item: 5, target: -1 }, 0)
    expect(fx.cast).toEqual({ seat: 0 })
    expect(fx.lunge).toBeNull()
  })

  it('no lunge/cast/splash on pass', () => {
    const prev = snap(player(30, []), player(30, []))
    const next = snap(player(30, []), player(30, []))
    expect(computeFx(prev, next, { t: 'pass' }, 0)).toEqual({
      lunge: null, cast: null, splashes: [],
    })
  })
})
