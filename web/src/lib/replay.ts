// web/src/lib/replay.ts
export type ActionDict =
  | { t: 'summon'; id: number }
  | { t: 'attack'; a: number; target: number }
  | { t: 'use'; item: number; target: number }
  | { t: 'pass' }

export interface CardState {
  iid: number; card_id: number; atk: number; def: number; abilities: string
  can_attack?: boolean; has_attacked?: boolean
}

export interface PlayerState {
  health: number; mana: number; max_mana: number; next_rune: number
  bonus_draw: number; deck_count: number; hand: CardState[]; board: CardState[]
}

export interface Snapshot { current: number; players: [PlayerState, PlayerState] }

export interface Step { seat: number; turn: number; action: ActionDict; state: Snapshot }

export interface ReplayHeader {
  replay_id: string; created_at: string; source: string; format: string
  engine_version: string; policy_a: string; policy_b: string; seed: number
  a_seat: number; winner: number; turns: number; step_count: number; hash: string
}

export interface Replay {
  header: ReplayHeader
  draft: { pool: number[][]; picks: { round: number; seat: number; pick: number }[] }
  battle: { opening: Snapshot; steps: Step[] }
  result: { winner: number; turns: number }
}

export interface Frame {
  index: number; snapshot: Snapshot; action: ActionDict | null
  seat: number | null; turn: number | null
}

export class Playback {
  frames: Frame[]
  cursor = 0
  cardIds = new Map<number, number>() // instance id -> catalog card_id

  constructor(replay: Replay) {
    const frames: Frame[] = [{
      index: 0, snapshot: replay.battle.opening, action: null, seat: null, turn: null,
    }]
    replay.battle.steps.forEach((s, i) => frames.push({
      index: i + 1, snapshot: s.state, action: s.action, seat: s.seat, turn: s.turn,
    }))
    this.frames = frames
    for (const f of this.frames) {
      for (const p of f.snapshot.players) {
        for (const c of p.board ?? []) this.cardIds.set(c.iid, c.card_id)
        for (const c of p.hand ?? []) this.cardIds.set(c.iid, c.card_id)
      }
    }
  }

  get current(): Frame { return this.frames[this.cursor] }

  seek(i: number): void {
    this.cursor = Math.max(0, Math.min(this.frames.length - 1, i))
  }

  next(): void { this.seek(this.cursor + 1) }
  prev(): void { this.seek(this.cursor - 1) }

  nextTurn(): void {
    const t = this.current.turn
    for (let i = this.cursor + 1; i < this.frames.length; i++) {
      if (this.frames[i].turn !== t) { this.cursor = i; return }
    }
    this.cursor = this.frames.length - 1
  }

  prevTurn(): void {
    const t = this.current.turn
    for (let i = this.cursor - 1; i >= 0; i--) {
      if (this.frames[i].turn !== t) { this.cursor = i; return }
    }
    this.cursor = 0
  }
}
