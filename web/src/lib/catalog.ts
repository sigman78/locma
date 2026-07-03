// Cached policy catalog: baseline names + spec templates + pinned depot models.
import { getPolicyCatalog, type PolicyCatalog } from './api'

let cached: Promise<PolicyCatalog> | null = null

export function policyCatalog(): Promise<PolicyCatalog> {
  cached ??= getPolicyCatalog().catch((e) => {
    cached = null // retry on next call
    throw e
  })
  return cached
}
