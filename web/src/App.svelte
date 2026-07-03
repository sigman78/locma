<!-- web/src/App.svelte — the consolidated panel: Experiments | Depot | Replays | Play -->
<script lang="ts">
  import { onMount } from 'svelte'
  import Experiments from './components/Experiments/Experiments.svelte'
  import Depot from './components/Depot/Depot.svelte'
  import Replays from './components/Replays/Replays.svelte'
  import Play from './components/Play/Play.svelte'
  import { loadCards } from './lib/cards'

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

  function fromHash(): TabId {
    const h = location.hash.replace(/^#\/?/, '')
    return (TABS.some((t) => t.id === h) ? h : 'experiments') as TabId
  }
  function select(id: TabId) {
    location.hash = `#/${id}`
  }

  onMount(() => {
    tab = fromHash()
    const onHash = () => (tab = fromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  })

  loadCards().then(() => (ready = true)).catch((e) => (error = String(e)))
</script>

<div class="shell">
  <nav>
    <span class="brand">LOCM panel</span>
    {#each TABS as t}
      <button class:active={tab === t.id} on:click={() => select(t.id)}>{t.label}</button>
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
    <section class:hidden={tab !== 'replays'}><Replays /></section>
    <section class:hidden={tab !== 'play'}><Play /></section>
  {/if}
</div>

<style>
  :global(body) { margin: 0; background: #0e0e12; font-family: system-ui, sans-serif; }
  :global(#app) { width: 100%; max-width: none; }
  .shell { color: #ddd; min-height: 100vh; }
  nav { display: flex; align-items: center; gap: 4px; padding: 8px 16px;
    border-bottom: 1px solid #23232b; background: #121218; position: sticky; top: 0; z-index: 50; }
  .brand { font-weight: 700; font-size: 14px; color: #8f8fa8; margin-right: 16px;
    letter-spacing: 0.06em; text-transform: uppercase; }
  nav button { background: none; border: none; color: #9a9ab0; font-size: 14px; padding: 8px 14px;
    cursor: pointer; border-radius: 6px; }
  nav button:hover { color: #ddd; background: #1c1c24; }
  nav button.active { color: #fff; background: #23233a; }
  section { padding: 12px 16px; }
  section.hidden { display: none; }
  .error { color: #ff6b6b; padding: 16px; }
  .loading { padding: 16px; color: #888; }
  .error button { background: #23232b; color: #ddd; border: 1px solid #333;
    border-radius: 4px; padding: 3px 10px; cursor: pointer; margin-left: 8px; }
</style>
