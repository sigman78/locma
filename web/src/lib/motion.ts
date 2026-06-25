import { get, writable } from 'svelte/store'
import { backOut } from 'svelte/easing'

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

/** Summon: overshoot spring into the slot — only inside the forward window. */
export function spring(_node: HTMLElement) {
  if (!get(animate)) return { duration: 0 }
  return {
    duration: 320,
    easing: backOut,
    css: (t: number) => `opacity:${Math.min(1, t * 1.4)}; transform:scale(${0.6 + 0.4 * t}) translateY(${(1 - t) * 16}px);`,
  }
}

/** Deal/draw: slide in from the right with a rotateY edge-flip reveal. */
export function dealIn(_node: HTMLElement) {
  if (!get(animate)) return { duration: 0 }
  return {
    duration: 300,
    css: (t: number) => `opacity:${Math.min(1, t * 1.5)}; transform:translateX(${(1 - t) * 60}px) rotateY(${(1 - t) * 92}deg);`,
  }
}
