import type { ActionDict, CardState, EventDict } from './replay'

export interface SlideFx {
  iid: number
  dx: number
  dy: number
}
export interface DyingFx {
  seat: number
  iid: number
  amount: number
}
export interface StepFx {
  slides: SlideFx[]
  flashes: (number | 'face')[]
  dying: DyingFx[]
}
export type RectOf = (key: number | 'face') => { cx: number; cy: number } | null

/**
 * Plan the visual effects for one applied action + its events.
 * `rectOf` returns screen centers from the live anchor registry (or null).
 * `fallbackDy` is the vertical step an attacker takes toward the opponent
 * (negative for the human's bottom row, positive for the opponent's top row)
 * — used only when the target's rect is unavailable.
 */
export function planStepFx(
  action: ActionDict | null,
  events: EventDict[],
  rectOf: RectOf,
  fallbackDy: number,
): StepFx {
  const slides: SlideFx[] = []
  const flashes: (number | 'face')[] = []
  const dying: DyingFx[] = []

  const fatalAmt = new Map<number, number>()
  for (const e of events) {
    if (e.t === 'damage' && e.fatal && typeof e.target === 'number') fatalAmt.set(e.target, e.amount)
  }
  for (const e of events) {
    if (e.t === 'unit_died') dying.push({ seat: e.seat, iid: e.iid, amount: fatalAmt.get(e.iid) ?? 0 })
  }

  if (action?.t === 'attack') {
    const from = rectOf(action.a)
    const toKey: number | 'face' = action.target === -1 ? 'face' : action.target
    const to = rectOf(toKey)
    if (from && to) {
      slides.push({ iid: action.a, dx: to.cx - from.cx, dy: to.cy - from.cy })
    } else if (from) {
      slides.push({ iid: action.a, dx: 0, dy: fallbackDy })
    }
  } else if (action?.t === 'use') {
    if (action.target >= 0) flashes.push(action.target)
    else if (events.some((e) => e.t === 'damage' && e.target === 'face')) flashes.push('face')
  }

  return { slides, flashes, dying }
}

/** The board to render = the settled view board plus any retained dying cards. */
export function mergeDisplayBoard(viewBoard: CardState[], dyingCards: CardState[]): CardState[] {
  const present = new Set(viewBoard.map((c) => c.iid))
  return [...viewBoard, ...dyingCards.filter((c) => !present.has(c.iid))]
}
