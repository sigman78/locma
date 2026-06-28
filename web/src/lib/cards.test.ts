import { describe, it, expect } from 'vitest'
import { stripItemPreface, spellEffectText, creatureSpecial } from './cards'
import type { CardMeta } from './api'

const meta = (over: Partial<CardMeta> = {}): CardMeta => ({
  id: 1, name: 'x', type: 'itemred', cost: 1, attack: 0, defense: 0,
  abilities: '------', player_hp: 0, enemy_hp: 0, card_draw: 0, description: '', ...over,
})

describe('stripItemPreface', () => {
  it('strips the colour-item preface + trailing separator', () => {
    expect(stripItemPreface('Green item. Give +2/+2.')).toBe('Give +2/+2.')
    expect(stripItemPreface('Red item: deal 3')).toBe('deal 3')
    expect(stripItemPreface('Blue item - draw a card')).toBe('draw a card')
  })
  it('leaves non-preface text untouched', () => {
    expect(stripItemPreface('Deal 2 damage.')).toBe('Deal 2 damage.')
    expect(stripItemPreface('')).toBe('')
  })
})

describe('spellEffectText', () => {
  it('returns the cleaned description when present', () => {
    expect(spellEffectText(meta({ description: 'Red item. Deal 3 damage.' }))).toBe('Deal 3 damage.')
  })
  it('derives a stat/HP/draw summary when there is no description', () => {
    expect(spellEffectText(meta({ attack: 2, defense: 1 }))).toBe('+2/+1')
    expect(spellEffectText(meta({ defense: -3 }))).toBe('0/-3')
    expect(spellEffectText(meta({ player_hp: 3, card_draw: 1 }))).toBe('+3♥ · draw +1')
    expect(spellEffectText(meta({ enemy_hp: -2 }))).toBe('foe -2♥')
  })
  it('falls back to the derived summary when the description is only the preface', () => {
    expect(spellEffectText(meta({ description: 'Green item.', attack: 1, defense: 1 }))).toBe('+1/+1')
  })
  it('returns empty for undefined meta', () => {
    expect(spellEffectText(undefined)).toBe('')
  })
})

describe('creatureSpecial', () => {
  it('drops the "X/Y Creature." preface, leaving the special', () => {
    expect(creatureSpecial('2/1 Creature. Summon: You gain 1 health.')).toBe('Summon: You gain 1 health.')
    expect(creatureSpecial('1/2 Creature. Summon: Deal 1 damage to your opponent.')).toBe('Summon: Deal 1 damage to your opponent.')
  })
  it('is empty for a vanilla creature or a keyword-only creature', () => {
    expect(creatureSpecial('2/2 Creature.')).toBe('')
    expect(creatureSpecial('2/2 Creature. Ward.')).toBe('')
  })
  it('drops bare keyword sentences but keeps the special', () => {
    expect(creatureSpecial('4/3 Creature. Charge. Summon: Deal 2 damage.')).toBe('Summon: Deal 2 damage.')
  })
  it('keeps a multi-sentence special', () => {
    expect(creatureSpecial('3/3 Creature. Summon: Draw a card. Gain 1 health.')).toBe('Summon: Draw a card. Gain 1 health.')
  })
  it('drops comma-separated keyword sentences (real cards)', () => {
    // Blizzard Demon: keywords only, no special → empty (no pill, no bottom text)
    expect(creatureSpecial('2/2 Creature. Charge, Drain.')).toBe('')
    // Night Howler: comma keywords + a special → keep only the special
    expect(creatureSpecial('6/5 Creature. Breakthrough, Drain. Summon: You lose 3 health.')).toBe(
      'Summon: You lose 3 health.',
    )
  })
})
