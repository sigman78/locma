import { computeFx, type Splash } from './fx'
import type { ActionDict, CardState, EventDict } from './replay'

export interface PlayView {
  turn: number
  me: {
    health: number
    mana: number
    max_mana: number
    deck_count: number
    bonus_draw: number
    hand: CardState[]
    board: CardState[]
  }
  op: {
    health: number
    mana: number
    max_mana: number
    deck_count: number
    bonus_draw: number
    hand_count: number
    board: CardState[]
  }
}

export interface DraftPending {
  phase: 'draft'
  you: number
  round: number
  total: number
  triplet: number[]
  my_picks: number
}

export interface BattlePending {
  phase: 'battle'
  you: number
  view: PlayView
  legal: ActionDict[]
}

export type Pending = DraftPending | BattlePending

export interface GameResult {
  winner_is_human: boolean
  turns: number
  replay_id: string
}

export interface Slice {
  events: EventDict[]
}

export interface GameSnapshot {
  status: 'awaiting_human' | 'finished'
  pending: Pending | null
  result: GameResult | null
}

export interface CreatedGame extends GameSnapshot {
  game_id: string
  you: number
}

export interface SubmitResponse extends GameSnapshot {
  slice: Slice
}

// --- legal-action predicates (drive what is clickable) ---
export const canSummon = (legal: ActionDict[], iid: number): boolean =>
  legal.some((a) => a.t === 'summon' && a.id === iid)

export const attackTargets = (legal: ActionDict[], aid: number): number[] =>
  legal
    .filter((a): a is Extract<ActionDict, { t: 'attack' }> => a.t === 'attack' && a.a === aid)
    .map((a) => a.target)

export const itemTargets = (legal: ActionDict[], iid: number): number[] =>
  legal
    .filter((a): a is Extract<ActionDict, { t: 'use' }> => a.t === 'use' && a.item === iid)
    .map((a) => a.target)

export const canPass = (legal: ActionDict[]): boolean => legal.some((a) => a.t === 'pass')

// --- splash lookup for damage numbers (events carry physical seat 0/1) ---
export const splashesFor = (events: EventDict[]): Splash[] =>
  computeFx(events, null, 0).splashes

export const cardDamage = (sp: Splash[], seat: number, iid: number): number | null => {
  const s = sp.find((x) => x.seat === seat && x.target === iid && !x.fatal)
  return s ? s.amount : null
}

export const faceDamage = (sp: Splash[], seat: number): number | null => {
  const s = sp.find((x) => x.target === 'face' && x.seat === seat)
  return s ? s.amount : null
}
