// web/src/lib/replay.test.ts
import { describe, expect, it } from 'vitest'
import { Playback, type Replay } from './replay'

const snap = (cur = 0): any => ({ current: cur, players: [{}, {}] })

// Steps carry the DECISION-POINT (pre-action) state, so state.current === seat for
// every step. A whole player-turn shares one turn number; `closing` is the final
// board after the game-ending action.
const replay: Replay = {
  header: {} as any,
  draft: { pool: [], picks: [] },
  battle: {
    opening: snap(0),
    steps: [
      { seat: 0, turn: 1, action: { t: 'summon', id: 10 }, state: snap(0), events: [] },
      { seat: 0, turn: 1, action: { t: 'pass' }, state: snap(0), events: [] },
      { seat: 1, turn: 2, action: { t: 'pass' }, state: snap(1), events: [] },
      { seat: 0, turn: 3, action: { t: 'pass' }, state: snap(0), events: [] },
    ],
    closing: snap(1),
  },
  result: { winner: 0, turns: 3 },
}

describe('Playback', () => {
  it('builds opening + steps as frames', () => {
    const pb = new Playback(replay)
    expect(pb.frames.length).toBe(5)
    expect(pb.frames[0].action).toBeNull()
    expect(pb.frames[1].turn).toBe(1)
  })

  it('reconstructs result-state frames from decision-point data', () => {
    const pb = new Playback(replay)
    // A within-turn move shows its result = the next same-seat decision point.
    expect(pb.frames[1].snapshot).toBe(replay.battle.steps[1].state)
    // A turn-ending pass keeps the acting seat's own board (not the opponent's
    // freshly-drawn turn), so its perspective stays with the actor.
    expect(pb.frames[2].snapshot).toBe(replay.battle.steps[1].state)
    expect(pb.frames[2].snapshot.current).toBe(0)
    // The last move has no later decision point — show the final closing board.
    expect(pb.frames[4].snapshot).toBe(replay.battle.closing)
  })

  it('next/prev clamp at bounds', () => {
    const pb = new Playback(replay)
    pb.prev()
    expect(pb.cursor).toBe(0)
    for (let i = 0; i < 9; i++) pb.next()
    expect(pb.cursor).toBe(4)
  })

  it('seek clamps', () => {
    const pb = new Playback(replay)
    pb.seek(99)
    expect(pb.cursor).toBe(4)
    pb.seek(-5)
    expect(pb.cursor).toBe(0)
  })

  it('nextTurn jumps to the next differing turn', () => {
    const pb = new Playback(replay)
    pb.seek(1) // turn 1
    pb.nextTurn()
    expect(pb.current.turn).toBe(2)
    pb.prevTurn()
    expect(pb.current.turn).toBe(1)
  })

  it('turn jumps clamp at bounds', () => {
    const pb = new Playback(replay)
    pb.seek(4) // last frame (turn 3)
    pb.nextTurn()
    expect(pb.cursor).toBe(4) // no later turn → stays at last frame
    pb.seek(0) // opening frame (turn null)
    pb.prevTurn()
    expect(pb.cursor).toBe(0) // no earlier turn → stays at frame 0
  })

  it('maps instance ids to catalog card ids across frames', () => {
    const withCard = (iid: number, cardId: number): any => ({
      current: 0,
      players: [
        { board: [{ iid, card_id: cardId }], hand: [] },
        { board: [], hand: [{ iid: iid + 1, card_id: cardId + 1 }] },
      ],
    })
    const rep: Replay = {
      header: {} as any,
      draft: { pool: [], picks: [] },
      battle: {
        opening: withCard(58, 48),
        steps: [{ seat: 0, turn: 1, action: { t: 'pass' }, state: withCard(58, 48), events: [] }],
      },
      result: { winner: 0, turns: 1 },
    }
    const pb = new Playback(rep)
    expect(pb.cardIds.get(58)).toBe(48)
    expect(pb.cardIds.get(59)).toBe(49)
    expect(pb.cardIds.get(999)).toBeUndefined()
  })
})
