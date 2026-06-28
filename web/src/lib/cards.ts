import { getCards, artUrl as apiArtUrl, type CardMeta } from './api'

let _cache: Map<number, CardMeta> | null = null
let _promise: Promise<Map<number, CardMeta>> | null = null

export function loadCards(): Promise<Map<number, CardMeta>> {
  return (_promise ??= getCards().then((list) => {
    _cache = new Map(list.map((c) => [c.id, c]))
    return _cache
  }))
}

export function card(id: number): CardMeta | undefined {
  return _cache?.get(id)
}

export function cardName(id: number): string {
  return _cache?.get(id)?.name ?? `#${id}`
}

export const artUrl = apiArtUrl

const sgn = (n: number) => (n > 0 ? `+${n}` : `${n}`)

/** Strip the redundant "Green/Red/Blue item" preface from a card description
 *  (the spell panel already encodes the colour). No-op for any other text. */
export function stripItemPreface(desc: string): string {
  return desc.replace(/^\s*(?:green|red|blue)\s+item\b[\s.:;,–-]*/i, '')
}

/** Compact spell-effect text for an item card: the cleaned printed description, else a
 *  derived stat/HP/draw summary. Empty for unknown cards / no effect. */
export function spellEffectText(meta: CardMeta | undefined): string {
  if (!meta) return ''
  const cleaned = stripItemPreface(meta.description)
  if (cleaned) return cleaned
  return [
    meta.attack || meta.defense ? `${sgn(meta.attack)}/${sgn(meta.defense)}` : '',
    meta.player_hp ? `${sgn(meta.player_hp)}♥` : '',
    meta.enemy_hp ? `foe ${sgn(meta.enemy_hp)}♥` : '',
    meta.card_draw ? `draw ${sgn(meta.card_draw)}` : '',
  ]
    .filter(Boolean)
    .join(' · ')
}
