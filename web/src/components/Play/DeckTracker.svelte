<!-- web/src/components/Play/DeckTracker.svelte -->
<!-- Minimalist deck tracker: the human's drafted deck, deduplicated and sorted
     by mana/name, one compact [◆cost name ×N] row each. A row grays out once all
     its copies have left the deck (been drawn), and carries a faint tint of its
     dominant keyword colour. Hovering a row reveals the full card. -->
<script lang="ts">
  import type { PlayView } from '../../lib/play'
  import type { CardState } from '../../lib/replay'
  import { card as cardMeta } from '../../lib/cards'
  import { abilityList } from '../../lib/abilities'
  import CardView from '../ReplayViewer/CardView.svelte'

  export let deck: number[] = [] // the 30 drafted card ids (this player's whole deck)
  export let view: PlayView // live battle view — its me.hand / me.board reveal draws

  // item cards carry no keyword mask; their identity colour is the item kind
  const ITEM_COLORS: Record<string, string> = {
    itemgreen: '#4fd97a',
    itemred: '#ff5d5d',
    itemblue: '#5aa9ff',
  }

  // the row's subtle tint: an item's colour, else its first (major) keyword colour
  function dominantColor(id: number): string | null {
    const m = cardMeta(id)
    if (!m) return null
    if (m.type in ITEM_COLORS) return ITEM_COLORS[m.type]
    return abilityList(m.abilities)[0]?.color ?? null
  }

  function toCard(id: number): CardState {
    const m = cardMeta(id)
    return {
      iid: -1,
      card_id: id,
      atk: m?.attack ?? 0,
      def: m?.defense ?? 0,
      abilities: m?.abilities ?? '',
    }
  }

  // Cumulative record of every card instance we've watched leave the deck: once an
  // iid shows up in hand or on the board it's been drawn, and it stays "drawn" even
  // after it's later played, discarded, or killed (it never returns to the deck).
  let seen = new Map<number, number>() // iid -> card_id
  let seenVersion = 0
  let deckRef: number[] | null = null

  function ingest(v: PlayView) {
    let changed = false
    for (const c of v.me.hand) if (!seen.has(c.iid)) { seen.set(c.iid, c.card_id); changed = true }
    for (const c of v.me.board) if (!seen.has(c.iid)) { seen.set(c.iid, c.card_id); changed = true }
    if (changed) seenVersion++
  }

  // one block so the deck-swap reset always runs before that view's ingest
  $: if (view) {
    if (deck !== deckRef) { deckRef = deck; seen = new Map(); seenVersion++ }
    ingest(view)
  }

  // seenVersion is an explicit dependency so the tally recomputes on each new draw
  function tally(s: Map<number, number>, _v: number): Map<number, number> {
    const m = new Map<number, number>()
    for (const cid of s.values()) m.set(cid, (m.get(cid) ?? 0) + 1)
    return m
  }
  $: drawnByCard = tally(seen, seenVersion)

  interface Row {
    id: number
    copies: number
    drawn: number
    color: string | null
    cost: number
    name: string
  }

  $: rows = ((): Row[] => {
    const copies = new Map<number, number>()
    for (const id of deck) copies.set(id, (copies.get(id) ?? 0) + 1)
    const out: Row[] = []
    for (const [id, n] of copies) {
      const m = cardMeta(id)
      out.push({
        id,
        copies: n,
        drawn: Math.min(n, drawnByCard.get(id) ?? 0),
        color: dominantColor(id),
        cost: m?.cost ?? 0,
        name: m?.name ?? `#${id}`,
      })
    }
    out.sort((a, b) => a.cost - b.cost || a.name.localeCompare(b.name))
    return out
  })()

  $: remaining = rows.reduce((s, r) => s + (r.copies - r.drawn), 0)
</script>

{#if rows.length}
  <aside class="tracker">
    <div class="head">Deck <span class="left">{remaining}</span></div>
    <ul>
      {#each rows as r (r.id)}
        <li
          class="row"
          class:depleted={r.drawn >= r.copies}
          style={r.color ? `--kw:${r.color}` : ''}>
          <span class="pip">{r.cost}</span>
          <span class="nm">{r.name}</span>
          {#if r.copies > 1}<span class="cnt">×{r.copies}</span>{/if}
          <div class="preview"><CardView card={toCard(r.id)} /></div>
        </li>
      {/each}
    </ul>
  </aside>
{/if}

<style>
  .tracker {
    width: 190px;
    background: #15151b;
    border: 1px solid #26262f;
    border-radius: 8px;
    padding: 8px;
    color: #ddd;
    font-size: 12px;
  }
  .head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-weight: 700;
    color: #9aa0ff;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 11px;
    margin-bottom: 6px;
    padding: 0 2px;
  }
  .head .left { color: #ffd23d; font-size: 13px; }
  ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
  .row {
    position: relative;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 6px;
    border-radius: 4px;
    border-left: 3px solid var(--kw, #3a3f55);
    /* faint keyword tint; neutral when the card has no keyword colour */
    background: color-mix(in srgb, var(--kw, #2a2a34) 14%, #1b1b22);
    cursor: default;
    transition: filter 0.12s ease, background 0.12s ease;
  }
  .row:hover { background: color-mix(in srgb, var(--kw, #2a2a34) 26%, #1b1b22); }
  /* all copies drawn: fade the row to gray so what's left in the deck stands out */
  .row.depleted { filter: grayscale(1); opacity: 0.4; }
  .row.depleted .nm { text-decoration: line-through; }
  .pip {
    flex: none;
    min-width: 18px;
    text-align: center;
    font-weight: 700;
    font-size: 11px;
    line-height: 16px;
    color: #cfe6ff;
    background: rgba(8, 12, 24, 0.85);
    border: 1px solid #5a7fd0;
    border-radius: 3px;
  }
  .nm {
    flex: 1 1 auto;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .cnt { flex: none; color: #ffd23d; font-weight: 700; }

  /* full-card reveal on hover — floats to the left of the panel (toward the board)
     so it never spills off the right edge of the screen */
  .preview {
    --card-w: 108px;
    --card-h: 150px;
    position: absolute;
    right: calc(100% + 10px);
    top: 50%;
    transform: translateY(-50%);
    width: var(--card-w);
    height: var(--card-h);
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    z-index: 200;
    filter: drop-shadow(0 10px 24px rgba(0, 0, 0, 0.7));
    transition: opacity 0.1s ease;
  }
  .row:hover .preview { opacity: 1; visibility: visible; }
</style>
