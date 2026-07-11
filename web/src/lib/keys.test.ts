import { describe, expect, it } from 'vitest'
import { digitIndex, isTypingTarget } from './keys'

// The suite runs in the `node` environment (no DOM), so we hand isTypingTarget
// minimal stand-ins with just the fields it reads.
const stub = (tagName: string, isContentEditable = false) =>
  ({ tagName, isContentEditable }) as unknown as EventTarget

describe('isTypingTarget', () => {
  it('is false for null / non-elements', () => {
    expect(isTypingTarget(null)).toBe(false)
    expect(isTypingTarget({} as EventTarget)).toBe(false)
  })
  it('is true for form fields (any case)', () => {
    for (const tag of ['INPUT', 'textarea', 'Select']) {
      expect(isTypingTarget(stub(tag))).toBe(true)
    }
  })
  it('is true for contenteditable, false for a plain button', () => {
    expect(isTypingTarget(stub('DIV', true))).toBe(true)
    expect(isTypingTarget(stub('BUTTON'))).toBe(false)
  })
})

describe('digitIndex', () => {
  it('maps 1..N to 0-based indices within range', () => {
    expect(digitIndex('1', 4)).toBe(0)
    expect(digitIndex('4', 4)).toBe(3)
  })
  it('returns null past the count', () => {
    expect(digitIndex('5', 4)).toBe(null)
  })
  it('rejects non-digits and 0', () => {
    expect(digitIndex('0', 4)).toBe(null)
    expect(digitIndex('a', 4)).toBe(null)
    expect(digitIndex('12', 4)).toBe(null)
  })
})
