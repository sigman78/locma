import { describe, expect, it } from 'vitest'
import type { ActionDict, CardState, EventDict } from './replay'
import { mergeDisplayBoard, planStepFx, type RectOf } from './stepfx'

const rects: Record<string, { cx: number; cy: number }> = {
  '7': { cx: 100, cy: 500 }, // attacker (my board, bottom)
  '12': { cx: 300, cy: 100 }, // a target (op board, top)
  face: { cx: 320, cy: 40 },
}
const rectOf: RectOf = (k) => rects[String(k)] ?? null

describe('planStepFx — attack slide', () => {
  it('measures the vector from attacker to a board target', () => {
    const a: ActionDict = { t: 'attack', a: 7, target: 12 }
    const p = planStepFx(a, [], rectOf, -40)
    expect(p.slides).toEqual([{ iid: 7, dx: 200, dy: -400 }])
  })
  it('slides toward the face anchor for a face attack', () => {
    const a: ActionDict = { t: 'attack', a: 7, target: -1 }
    const p = planStepFx(a, [], rectOf, -40)
    expect(p.slides).toEqual([{ iid: 7, dx: 220, dy: -460 }])
  })
  it('falls back to a vertical step when the target rect is missing', () => {
    const a: ActionDict = { t: 'attack', a: 7, target: 999 } // no rect for 999
    const p = planStepFx(a, [], rectOf, -40)
    expect(p.slides).toEqual([{ iid: 7, dx: 0, dy: -40 }])
  })
  it('produces no slide when the attacker rect is missing', () => {
    const a: ActionDict = { t: 'attack', a: 555, target: 12 } // no rect for 555
    expect(planStepFx(a, [], rectOf, -40).slides).toEqual([])
  })
})

describe('planStepFx — flash + dying', () => {
  it('flashes a targeted item victim', () => {
    const a: ActionDict = { t: 'use', item: 3, target: 12 }
    expect(planStepFx(a, [], rectOf, 40).flashes).toEqual([12])
  })
  it('flashes the face when a no-target item damages the face', () => {
    const a: ActionDict = { t: 'use', item: 3, target: -1 }
    const evs: EventDict[] = [{ t: 'damage', seat: 0, target: 'face', amount: 3, fatal: false }]
    expect(planStepFx(a, evs, rectOf, 40).flashes).toEqual(['face'])
  })
  it('derives dying units with the fatal damage amount', () => {
    const evs: EventDict[] = [
      { t: 'damage', seat: 1, target: 12, amount: 5, fatal: true },
      { t: 'unit_died', seat: 1, iid: 12 },
    ]
    expect(planStepFx({ t: 'attack', a: 7, target: 12 }, evs, rectOf, -40).dying).toEqual([
      { seat: 1, iid: 12, amount: 5 },
    ])
  })
  it('reports a death with amount 0 when there is no fatal damage event (item kill)', () => {
    const evs: EventDict[] = [{ t: 'unit_died', seat: 1, iid: 9 }]
    expect(planStepFx({ t: 'use', item: 3, target: 9 }, evs, rectOf, 40).dying).toEqual([
      { seat: 1, iid: 9, amount: 0 },
    ])
  })
  it('does not flash when a no-target item has no face damage', () => {
    const a: ActionDict = { t: 'use', item: 3, target: -1 }
    expect(planStepFx(a, [], rectOf, 40).flashes).toEqual([])
  })
})

describe('mergeDisplayBoard', () => {
  const c = (iid: number): CardState => ({ iid, card_id: 1, atk: 1, def: 1, abilities: '' })
  it('appends dying cards not present in the view board', () => {
    expect(mergeDisplayBoard([c(1)], [c(2)]).map((x) => x.iid)).toEqual([1, 2])
  })
  it('does not duplicate a card already in the view board', () => {
    expect(mergeDisplayBoard([c(1), c(2)], [c(2)]).map((x) => x.iid)).toEqual([1, 2])
  })
  it('returns the view board unchanged when nothing is dying', () => {
    expect(mergeDisplayBoard([c(1)], []).map((x) => x.iid)).toEqual([1])
  })
})
