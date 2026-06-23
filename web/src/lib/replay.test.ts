// web/src/lib/replay.test.ts
import { describe, expect, it } from 'vitest'
import { Playback, type Replay } from './replay'

const snap = (cur = 0): any => ({ current: cur, players: [{}, {}] })

const replay: Replay = {
  header: {} as any,
  draft: { pool: [], picks: [] },
  battle: {
    opening: snap(0),
    steps: [
      { seat: 0, turn: 1, action: { t: 'pass' }, state: snap(1) },
      { seat: 1, turn: 2, action: { t: 'pass' }, state: snap(0) },
      { seat: 0, turn: 3, action: { t: 'pass' }, state: snap(1) },
    ],
  },
  result: { winner: 0, turns: 3 },
}

describe('Playback', () => {
  it('builds opening + steps as frames', () => {
    const pb = new Playback(replay)
    expect(pb.frames.length).toBe(4)
    expect(pb.frames[0].action).toBeNull()
    expect(pb.frames[1].turn).toBe(1)
  })

  it('next/prev clamp at bounds', () => {
    const pb = new Playback(replay)
    pb.prev()
    expect(pb.cursor).toBe(0)
    pb.next(); pb.next(); pb.next(); pb.next()
    expect(pb.cursor).toBe(3)
  })

  it('seek clamps', () => {
    const pb = new Playback(replay)
    pb.seek(99)
    expect(pb.cursor).toBe(3)
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
    pb.seek(3) // last frame (turn 3)
    pb.nextTurn()
    expect(pb.cursor).toBe(3) // no later turn → stays at last frame
    pb.seek(0) // opening frame (turn null)
    pb.prevTurn()
    expect(pb.cursor).toBe(0) // no earlier turn → stays at frame 0
  })
})
