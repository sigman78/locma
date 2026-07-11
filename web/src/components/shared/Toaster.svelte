<!-- Fixed, non-blocking notification stack. Mounted once at the app shell. -->
<script lang="ts">
  import { fly, fade } from 'svelte/transition'
  import { dismissToast, toasts } from '../../lib/toast'
</script>

<div class="toaster" aria-live="polite" aria-atomic="false">
  {#each $toasts as t (t.id)}
    <div
      class="toast {t.kind}"
      role={t.kind === 'error' ? 'alert' : 'status'}
      in:fly={{ y: 12, duration: 180 }}
      out:fade={{ duration: 160 }}
    >
      <span class="msg">{t.message}</span>
      <button class="x" title="dismiss" on:click={() => dismissToast(t.id)}>x</button>
    </div>
  {/each}
</div>

<style>
  .toaster {
    position: fixed;
    right: 16px;
    bottom: 16px;
    z-index: 2000;
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-width: min(420px, calc(100vw - 32px));
    pointer-events: none;
  }
  .toast {
    pointer-events: auto;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #1b1b24;
    border: 1px solid #33333f;
    border-left: 3px solid #4fa3d9;
    border-radius: 6px;
    padding: 10px 12px;
    color: #ddd;
    font-size: 13px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
  }
  .toast.error { border-left-color: #ff6b6b; background: #241519; }
  .msg { flex: 1; word-break: break-word; white-space: pre-wrap; }
  .x {
    background: none;
    border: none;
    color: #888;
    cursor: pointer;
    font-size: 13px;
    line-height: 1;
    padding: 0 2px;
  }
  .x:hover { color: #ddd; }
</style>
