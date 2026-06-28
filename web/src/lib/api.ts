import type { Replay, ReplayHeader } from './replay'
import type { ActionDict } from './replay'
import type { CreatedGame, GameSnapshot, SubmitResponse } from './play'

export interface CardMeta {
  id: number; name: string; type: string; cost: number; attack: number
  defense: number; abilities: string; player_hp: number; enemy_hp: number
  card_draw: number; description: string; card_text: string
}

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}

export const getCards = () => fetch('/api/cards').then(j<CardMeta[]>)
export const getPolicies = () => fetch('/api/policies').then(j<string[]>)
export const listReplays = () => fetch('/api/replays').then(j<ReplayHeader[]>)
export const getReplay = (id: string) => fetch(`/api/replays/${id}`).then(j<Replay>)
export const listGameLogs = () =>
  fetch('/api/game-logs').then(j<{ path: string; rows: number }[]>)

export const runReplay = (body: { policy_a: string; policy_b: string; seed: number }) =>
  fetch('/api/replays', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  }).then(j<ReplayHeader>)

export const importReplay = (body: { path: string; row: number }) =>
  fetch('/api/replays/import', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  }).then(j<ReplayHeader>)

export const artUrl = (cardId: number) => `/api/art/${cardId}`

export const createGame = (body: { opponent: string; seed?: number }) =>
  fetch('/api/games', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  }).then(j<CreatedGame>)

export const getGame = (id: string) => fetch(`/api/games/${id}`).then(j<GameSnapshot>)

export const submitDraft = (id: string, pick: number) =>
  fetch(`/api/games/${id}/draft`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ pick }),
  }).then(j<SubmitResponse>)

export const submitAction = (id: string, action: ActionDict) =>
  fetch(`/api/games/${id}/action`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ action }),
  }).then(j<SubmitResponse>)
