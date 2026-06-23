import { getCards, artUrl as apiArtUrl, type CardMeta } from './api'

let _cache: Map<number, CardMeta> | null = null

export async function loadCards(): Promise<Map<number, CardMeta>> {
  if (_cache) return _cache
  const list = await getCards()
  _cache = new Map(list.map((c) => [c.id, c]))
  return _cache
}

export function card(id: number): CardMeta | undefined {
  return _cache?.get(id)
}

export function cardName(id: number): string {
  return _cache?.get(id)?.name ?? `#${id}`
}

export const artUrl = apiArtUrl
