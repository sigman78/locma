import type { ActionDict, CardState, Snapshot } from './replay'

export interface Splash {
  seat: number
  target: number | 'face' // minion iid, or 'face'
  amount: number
  fatal: boolean
}

export interface Lunge {
  seat: number
  iid: number
  toward: 'face' | { seat: number; iid: number }
}

export interface Cast {
  seat: number
}

export interface Fx {
  lunge: Lunge | null
  cast: Cast | null
  splashes: Splash[]
}

function boardSplashes(seat: number, before: CardState[], after: CardState[]): Splash[] {
  const out: Splash[] = []
  const byIid = new Map(after.map((c) => [c.iid, c]))
  for (const c of before) {
    const now = byIid.get(c.iid)
    if (!now) out.push({ seat, target: c.iid, amount: 0, fatal: true })
    else if (now.def < c.def) {
      out.push({ seat, target: c.iid, amount: c.def - now.def, fatal: false })
    }
  }
  return out
}

export function computeFx(
  prev: Snapshot,
  next: Snapshot,
  action: ActionDict | null,
  seat: number,
): Fx {
  const splashes: Splash[] = []
  for (let s = 0; s < 2; s++) {
    splashes.push(...boardSplashes(s, prev.players[s].board, next.players[s].board))
    const lost = prev.players[s].health - next.players[s].health
    if (lost > 0) splashes.push({ seat: s, target: 'face', amount: lost, fatal: false })
  }

  let lunge: Lunge | null = null
  let cast: Cast | null = null
  if (action?.t === 'attack') {
    const attacker = prev.players[seat].board.find((c) => c.iid === action.a)
    if (attacker) {
      lunge = {
        seat,
        iid: attacker.iid,
        toward: action.target === -1 ? 'face' : { seat: 1 - seat, iid: action.target },
      }
    }
  } else if (action?.t === 'use') {
    cast = { seat }
  }

  return { lunge, cast, splashes }
}
