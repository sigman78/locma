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
export const getArtIndex = () => fetch('/api/art-index').then(j<number[]>)

// -- policy catalog ---------------------------------------------------------

export interface PolicyCatalog {
  baselines: string[]
  model_bases: { base: string; template: string }[]
  depot_models: { name: string; version: number; refs: string[] }[]
  suggestions: string[]
}

export const getPolicyCatalog = () => fetch('/api/policy-catalog').then(j<PolicyCatalog>)

// -- experiments --------------------------------------------------------------

export interface SchemaField {
  name: string
  type: 'policy' | 'policies' | 'int' | 'float' | 'str'
  default: unknown
  help?: string
}

export interface ExperimentKind {
  kind: string
  label: string
  description: string
  schema: SchemaField[]
}

export interface Preset {
  id: string
  name: string
  kind: string
  params: Record<string, unknown>
  note: string
}

export interface ExpJob {
  job_id: string
  kind: string
  name: string
  params: Record<string, unknown>
  state: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  progress_done: number
  progress_total: number
  created: number
  started: number | null
  finished: number | null
  result: Record<string, unknown> | null
  error: string | null
}

const post = (url: string, body?: unknown) =>
  fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })

export const getExperimentKinds = () =>
  fetch('/api/experiments/kinds').then(j<ExperimentKind[]>)
export const listPresets = () => fetch('/api/experiments/presets').then(j<Preset[]>)
export const savePreset = (id: string, body: Omit<Preset, 'id'>) =>
  fetch(`/api/experiments/presets/${id}`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  }).then(j<Preset>)
export const deletePreset = (id: string) =>
  fetch(`/api/experiments/presets/${id}`, { method: 'DELETE' }).then(j<{ deleted: string }>)
export const runExperiment = (body: {
  kind: string
  params: Record<string, unknown>
  name?: string
}) => post('/api/experiments/run', body).then(j<ExpJob>)
export const listJobs = () => fetch('/api/experiments/jobs').then(j<ExpJob[]>)
export const cancelJob = (id: string) =>
  post(`/api/experiments/jobs/${id}/cancel`).then(j<{ cancelled: string }>)

export type SeriesMap = Record<string, [number, number][]>
export interface JobSeries {
  series: SeriesMap
  live: Record<string, unknown>
}
export const getJobSeries = (id: string) =>
  fetch(`/api/experiments/jobs/${id}/series`).then(j<JobSeries>)
export const getJobLog = (id: string) =>
  fetch(`/api/experiments/jobs/${id}/log`).then(j<{ log: string }>)

// -- depot --------------------------------------------------------------------

export interface DepotVersion {
  version: number
  created: string
  git_commit: string | null
  git_dirty: boolean
  command: string
  parents: string[]
  note: string
  meta: Record<string, unknown>
  files: Record<string, { sha256: string; size: number }>
  published?: string
  status: 'local' | 'partial' | 'missing'
  size: number
}

export interface DepotRecord {
  name: string
  kind: string
  pin: number
  versions: DepotVersion[]
}

export const listDepot = () => fetch('/api/depot').then(j<DepotRecord[]>)
export const depotRemote = () => fetch('/api/depot/remote').then(j<{ remote: string }>)
export const depotPin = (name: string, version: number) =>
  post(`/api/depot/${name}/pin`, { version }).then(j<DepotRecord>)
export const depotPull = (name: string, version?: number) =>
  post(`/api/depot/${name}/pull`, { version }).then(j<{ fetched: string[]; record: DepotRecord }>)
export const depotPush = (name: string, version?: number) =>
  post(`/api/depot/${name}/push`, { version }).then(j<{ locator: string; record: DepotRecord }>)
export const depotPublish = (body: {
  name: string
  files: string[]
  kind?: string
  note?: string
  parents?: string[]
  meta?: Record<string, unknown>
}) => post('/api/depot/publish', body).then(j<{ version: number; record: DepotRecord }>)
export const depotGc = (dry_run: boolean) =>
  post('/api/depot/gc', { dry_run }).then(j<{ removed: number; freed: number; dry_run: boolean }>)

// -- play ---------------------------------------------------------------------

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
