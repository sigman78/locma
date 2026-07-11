import { writable } from 'svelte/store'

export type ToastKind = 'info' | 'error'
export interface Toast {
  id: number
  kind: ToastKind
  message: string
}

// A tiny transient-notification stack. Non-blocking: use it for recoverable /
// informational messages. Anything that leaves game state uncertain should stay
// a deliberate modal, not a toast that auto-dismisses.
export const toasts = writable<Toast[]>([])

let seq = 0
const DEFAULT_MS = 5000

export function dismissToast(id: number): void {
  toasts.update((list) => list.filter((t) => t.id !== id))
}

export function pushToast(message: string, kind: ToastKind = 'info', ttlMs = DEFAULT_MS): number {
  const id = ++seq
  toasts.update((list) => [...list, { id, kind, message }])
  if (ttlMs > 0 && typeof setTimeout === 'function') {
    setTimeout(() => dismissToast(id), ttlMs)
  }
  return id
}

/** Convenience for `catch` blocks: coerce anything to a readable error toast. */
export function toastError(e: unknown): number {
  return pushToast(String(e), 'error', 8000)
}
