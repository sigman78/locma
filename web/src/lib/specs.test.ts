import { describe, expect, it } from 'vitest'
import { explainSpec, SPEC_INFO } from './specs'

describe('explainSpec', () => {
  it('explains a bare baseline', () => {
    const e = explainSpec('greedy')
    expect(e.known).toBe(true)
    expect(e.base).toBe('greedy')
    expect(e.params).toEqual([])
  })

  it('fills defaults for omitted positional params', () => {
    const e = explainSpec('vbeam:depot:b0/b0_s0.zip')
    expect(e.known).toBe(true)
    expect(e.params.map((p) => [p.name, p.value, p.isDefault])).toEqual([
      ['model', 'depot:b0/b0_s0.zip', false],
      ['width', '8', true],
      ['max_actions', '20', true],
    ])
  })

  it('keeps explicit params and marks them non-default', () => {
    const e = explainSpec('vbeam:runs/x.zip,12,30')
    expect(e.params.map((p) => p.value)).toEqual(['runs/x.zip', '12', '30'])
    expect(e.params.every((p) => !p.isDefault)).toBe(true)
  })

  it('splits on the FIRST colon only (depot refs contain colons)', () => {
    const e = explainSpec('ppo:depot:b0/b0_s2.zip')
    expect(e.base).toBe('ppo')
    expect(e.params[0].value).toBe('depot:b0/b0_s2.zip')
  })

  it('handles search specs with numeric params', () => {
    const e = explainSpec('dmcts:20,50')
    expect(e.params.map((p) => [p.name, p.value])).toEqual([
      ['K', '20'],
      ['I', '50'],
      ['seed', '0'],
      ['rollout_turns', '3'],
    ])
  })

  it('flags unknown bases', () => {
    expect(explainSpec('nope:1,2').known).toBe(false)
    expect(explainSpec('').known).toBe(false)
  })

  it('covers every documented base with a blurb', () => {
    for (const [base, info] of Object.entries(SPEC_INFO)) {
      expect(info.blurb.length, base).toBeGreaterThan(10)
    }
  })
})
