import { get } from 'svelte/store'
import { beforeEach, describe, expect, it } from 'vitest'
import { dismissToast, pushToast, toastError, toasts } from './toast'

beforeEach(() => toasts.set([]))

describe('toast store', () => {
  it('pushes and dismisses by id', () => {
    const id = pushToast('hello', 'info', 0)
    expect(get(toasts)).toHaveLength(1)
    expect(get(toasts)[0]).toMatchObject({ message: 'hello', kind: 'info' })
    dismissToast(id)
    expect(get(toasts)).toHaveLength(0)
  })

  it('assigns unique ids and keeps insertion order', () => {
    const a = pushToast('a', 'info', 0)
    const b = pushToast('b', 'info', 0)
    expect(a).not.toBe(b)
    expect(get(toasts).map((t) => t.message)).toEqual(['a', 'b'])
  })

  it('toastError stringifies and marks kind=error', () => {
    toastError(new Error('boom'))
    expect(get(toasts)[0]).toMatchObject({ kind: 'error' })
    expect(get(toasts)[0].message).toContain('boom')
  })

  it('auto-dismisses after the ttl', async () => {
    pushToast('fleeting', 'info', 10)
    expect(get(toasts)).toHaveLength(1)
    await new Promise((r) => setTimeout(r, 25))
    expect(get(toasts)).toHaveLength(0)
  })
})
