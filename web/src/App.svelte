<!-- web/src/App.svelte — the consolidated panel: Experiments | Depot | Replays | Play -->
<script lang="ts">
  import { onMount } from 'svelte'
  import Experiments from './components/Experiments/Experiments.svelte'
  import Depot from './components/Depot/Depot.svelte'
  import Replays from './components/Replays/Replays.svelte'
  import Play from './components/Play/Play.svelte'
  import Toaster from './components/shared/Toaster.svelte'
  import { loadCards } from './lib/cards'
  import { digitIndex, isTypingTarget } from './lib/keys'
  import { toastError } from './lib/toast'

  const TABS = [
    { id: 'experiments', label: 'Experiments' },
    { id: 'depot', label: 'Depot' },
    { id: 'replays', label: 'Replays' },
    { id: 'play', label: 'Play' },
  ] as const
  type TabId = (typeof TABS)[number]['id']

  let ready = false
  let error: string | null = null
  let tab: TabId = 'experiments'
  // deep link: '#/replays/<replay_id>' opens that replay directly (e.g. the
  // Play end-overlay's "View replay"); a bare '#/replays' shows the library
  let replayId: string | null = null

  function fromHash() {
    const h = location.hash.replace(/^#\/?/, '')
    const [head, ...rest] = h.split('/')
    tab = (TABS.some((t) => t.id === head) ? head : 'experiments') as TabId
    replayId = tab === 'replays' && rest.length ? decodeURIComponent(rest.join('/')) : null
  }
  function select(id: TabId) {
    location.hash = `#/${id}`
  }

  // Alt+1..N jump straight to a tab. Alt keeps them out of the way of the
  // play screens (which use bare digits for the draft) and of typing.
  function onKey(e: KeyboardEvent) {
    if (!e.altKey || e.ctrlKey || e.metaKey || isTypingTarget(e.target)) return
    const idx = digitIndex(e.key, TABS.length)
    if (idx !== null) {
      e.preventDefault()
      select(TABS[idx].id)
    }
  }

  onMount(() => {
    fromHash()
    window.addEventListener('hashchange', fromHash)
    return () => window.removeEventListener('hashchange', fromHash)
  })

  loadCards()
    .then(() => (ready = true))
    .catch((e) => {
      error = String(e)
      toastError(e)
    })
</script>

<svelte:window on:keydown={onKey} />

<div class="shell">
  <nav>
    <span class="brand">LOCM panel</span>
    {#each TABS as t, i}
      <button
        class:active={tab === t.id}
        title={`${t.label} (press Alt+${i + 1})`}
        on:click={() => select(t.id)}
      >
        <kbd>Alt+{i + 1}</kbd>{t.label}
      </button>
    {/each}
  </nav>

  {#if error}
    <p class="error">Error: {error} <button on:click={() => (error = null)}>dismiss</button></p>
  {:else if !ready}
    <p class="loading">loading cards…</p>
  {:else}
    <!-- keep tabs mounted so running jobs / an in-progress game survive tab switches -->
    <section class:hidden={tab !== 'experiments'}>
      <Experiments active={tab === 'experiments'} />
    </section>
    <section class:hidden={tab !== 'depot'}><Depot active={tab === 'depot'} /></section>
    <section class:hidden={tab !== 'replays'}>
      <Replays active={tab === 'replays'} openId={replayId} />
    </section>
    <section class:hidden={tab !== 'play'}><Play active={tab === 'play'} /></section>
  {/if}

  <Toaster />
</div>

<style>
  :global(body) { margin: 0; background: #0e0e12; font-family: system-ui, sans-serif; }
  :global(#app) { width: 100%; max-width: none; }
  .shell { color: #ddd; min-height: 100vh; }
  nav { display: flex; align-items: center; gap: 4px; padding: 8px 16px; flex-wrap: wrap;
    border-bottom: 1px solid #23232b; background: #121218; position: sticky; top: 0; z-index: 50; }
  .brand { font-weight: 700; font-size: 14px; color: #8f8fa8; margin-right: 16px;
    letter-spacing: 0.06em; text-transform: uppercase; }
  nav button { background: none; border: none; color: #9a9ab0; font-size: 14px; padding: 8px 14px;
    cursor: pointer; border-radius: 6px; }
  nav button:hover { color: #ddd; background: #1c1c24; }
  nav button.active { color: #fff; background: #23233a; }
  nav button kbd { font: inherit; font-size: 10px; color: #6b6b82; background: #0e0e14;
    border: 1px solid #2a2a36; border-radius: 3px; padding: 0 4px; margin-right: 7px;
    vertical-align: middle; }
  nav button.active kbd { color: #b7b7d6; border-color: #4a4f8a; }
  section { padding: 12px 16px; }
  section.hidden { display: none; }
  .error { color: #ff6b6b; padding: 16px; }
  .loading { padding: 16px; color: #888; }
  .error button { background: #23232b; color: #ddd; border: 1px solid #333;
    border-radius: 4px; padding: 3px 10px; cursor: pointer; margin-left: 8px; }
</style>
