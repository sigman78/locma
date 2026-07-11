// Small, pure keyboard helpers shared by the panel shell and the play screens.
// Kept side-effect-free so the mapping logic is unit-testable without a DOM.

/** True when a keystroke should be left to a focused text field / editable
 *  element rather than treated as a global shortcut. */
export function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null
  if (!el || !el.tagName) return false
  const tag = el.tagName.toLowerCase()
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return true
  return el.isContentEditable === true
}

/** Map a top-row digit key ('1'..'9') to a 0-based index, or null. `count`
 *  clamps the range so only real slots respond. */
export function digitIndex(key: string, count: number): number | null {
  if (key.length !== 1 || key < '1' || key > '9') return null
  const idx = key.charCodeAt(0) - '1'.charCodeAt(0)
  return idx < count ? idx : null
}
