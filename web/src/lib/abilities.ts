export const ABILITY_ORDER = 'BCDGLW'

export interface AbilityInfo {
  letter: string
  name: string
  color: string
}

export const ABILITIES: Record<string, AbilityInfo> = {
  B: { letter: 'B', name: 'Breakthrough', color: '#ff8a3d' },
  C: { letter: 'C', name: 'Charge', color: '#ffd23d' },
  D: { letter: 'D', name: 'Drain', color: '#c264ff' },
  G: { letter: 'G', name: 'Guard', color: '#5aa9ff' },
  L: { letter: 'L', name: 'Lethal', color: '#4fd97a' },
  W: { letter: 'W', name: 'Ward', color: '#7fe7ff' },
}

/** Aura keywords that get a prominent visual treatment. */
export const AURA_KEYWORDS = ['G', 'L', 'W'] as const

/** Present abilities of a 6-char BCDGLW mask, in mask order. */
export function abilityList(mask: string | null | undefined): AbilityInfo[] {
  if (!mask) return []
  return [...mask].filter((ch) => ch in ABILITIES).map((ch) => ABILITIES[ch])
}

/** True if the mask contains the given keyword letter. */
export function hasAura(mask: string | null | undefined, letter: string): boolean {
  return !!mask && mask.includes(letter)
}
