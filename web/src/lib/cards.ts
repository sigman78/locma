import { getCards, getArtIndex, artUrl as apiArtUrl, type CardMeta } from './api'

let _cache: Map<number, CardMeta> | null = null
let _art: Set<number> | null = null // null = index unavailable, assume art exists
let _promise: Promise<Map<number, CardMeta>> | null = null

export function loadCards(): Promise<Map<number, CardMeta>> {
  return (_promise ??= Promise.all([
    getCards(),
    getArtIndex().catch(() => null), // older server: fall back to per-image 404s
  ]).then(([list, art]) => {
    _cache = new Map(list.map((c) => [c.id, c]))
    _art = art === null ? null : new Set(art)
    return _cache
  }))
}

export function card(id: number): CardMeta | undefined {
  return _cache?.get(id)
}

export function cardName(id: number): string {
  return _cache?.get(id)?.name ?? `#${id}`
}

/** Whether a cached portrait exists — cards without one render the generated
 * placeholder directly, with no doomed image request. */
export function hasArt(id: number): boolean {
  return _art === null ? true : _art.has(id)
}

export const artUrl = apiArtUrl

// NOTE: a card's cleaned special/effect text is computed server-side in
// locma/data/cards_db.py and served as `CardMeta.card_text` (raw `description` kept too).
