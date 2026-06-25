# 0004 — Practicum is coupled to the observation/action encoding

## Status
Accepted (2026-06-25)

## Context
We distill cheating-MCTS play into a fast net by behavior cloning. Each training
example needs the student observation, the expert's action *index*, and the legal
mask. These are defined by `locma/envs/encode.py` (`encode_battle`, `action_mask`,
`ACTION_SIZE`). We considered storing only a thin `(seat, action)` trace and
deriving observations later via a trace-driven replay (decoupled from `encode.py`),
versus capturing the encoded example inline during the (single) generation run.

The encoding was originally positional (`ACTION_SIZE=64`, `OBS_SIZE=146`):
`index_to_action(idx, legal)` returned `legal[idx]` (position *idx* in the legal
list), and the mask was an information-free length prefix. That layout was replaced
with a fixed *semantic* action space (see `docs/ppo-review.md` for why the action
representation was the bottleneck).

The current encoding (`ACTION_SIZE=155`, `OBS_SIZE=308`) is slot-indexed:
- **Pass** — slot 0
- **Summon hand-slot s** — slots 1–8 (one per hand slot, s = 0–7)
- **Use hand-slot s → target** — slots 9–112 (8 hand slots × 13 target codes)
- **Attack board-slot a → target** — slots 113–154 (6 board slots × 7 target codes)

The observation (308-d) encodes 8 global scalars plus 20 card slots (hand 8 +
my board 6 + op board 6) × 15 features each. The mask flags exactly which
concrete actions are legal at each semantic slot.

## Decision
Capture inline during the MCTS generation run via the engine's `on_pre_step` hook.
The practicum stores already-encoded `obs`/`action`/`mask` arrays. The manifest
records `obs_size`/`action_size` (currently `obs_size=308`, `action_size=155`);
the distiller asserts they match the live `encode.py` and refuses a stale dataset,
so the coupling fails loudly rather than silently training on a mismatched layout.

## Consequences
- Simplest path: one generation pass, no trace-driven replay machinery.
- Changing the observation/action encoding invalidates existing practica; they
  must be regenerated (re-run MCTS). Any practicum generated under the old
  positional layout (`obs_size=146`, `action_size=64`) is stale and must be
  regenerated. The manifest guard makes the staleness explicit.
- A future decoupled `ReplayPolicy`-based derivation remains possible if encoding
  churn becomes painful.
