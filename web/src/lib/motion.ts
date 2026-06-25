import { get, writable } from 'svelte/store'

/** True only during a forward-step animation window; gates exit transitions. */
export const animate = writable(false)

let timer: ReturnType<typeof setTimeout> | null = null

/** Open the animation window, auto-closing after `ms`. */
export function pulse(ms = 450): void {
  animate.set(true)
  if (timer) clearTimeout(timer)
  timer = setTimeout(() => animate.set(false), ms)
}

/** Action that replays a CSS animation class whenever `token` changes. */
export function restartAnim(
  node: HTMLElement,
  params: { cls: string | null; token: number },
) {
  let current: string | null = null
  function run(p: { cls: string | null; token: number }) {
    if (current) node.classList.remove(current)
    current = p.cls
    if (!p.cls) return
    void node.offsetWidth // force reflow so the animation restarts
    node.classList.add(p.cls)
  }
  run(params)
  return { update: run }
}

/** Exit transition for a dying minion — only animates inside the forward window. */
export function deathFx(_node: HTMLElement) {
  if (!get(animate)) return { duration: 0 }
  return {
    duration: 280,
    css: (t: number) =>
      `opacity:${t}; transform:scale(${0.6 + 0.4 * t});` +
      `filter:saturate(${t}) brightness(${1 + (1 - t) * 1.5});`,
  }
}

/** Enter transition for a freshly-arrived card (summon/draw) — only inside the window. */
export function popIn(_node: HTMLElement) {
  if (!get(animate)) return { duration: 0 }
  return {
    duration: 240,
    css: (t: number) => `opacity:${t}; transform:scale(${0.7 + 0.3 * t});`,
  }
}
