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

/** A dying card retained for its cross/removal animation, with the board index
 *  it occupied before it died. */
export interface RetainedCard {
  card: CardState
  index: number
}

/**
 * The board to render = the settled view board with each retained dying card
 * re-inserted at its ORIGINAL slot index. Appending dying cards at the end (the
 * old behaviour) made a mid-row dying minion teleport to the rightmost slot
 * before its removal — and an attacker that dies on its own swing would then
 * slide from that wrong position. Inserting in ascending index order at the
 * original index keeps each one in place (each insert naturally precedes the
 * survivors that had a higher original index).
 */
export function mergeDisplayBoard(viewBoard: CardState[], dying: RetainedCard[]): CardState[] {
  const present = new Set(viewBoard.map((c) => c.iid))
  const out = [...viewBoard]
  for (const d of dying
    .filter((d) => !present.has(d.card.iid))
    .sort((a, b) => a.index - b.index)) {
    out.splice(Math.min(d.index, out.length), 0, d.card)
  }
  return out
}
