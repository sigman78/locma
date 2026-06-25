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

## Decision
Capture inline during the MCTS generation run via the engine's `on_pre_step` hook.
The practicum stores already-encoded `obs`/`action`/`mask` arrays. The manifest
records `obs_size`/`action_size`; the distiller asserts they match the live
`encode.py` and refuses a stale dataset, so the coupling fails loudly rather than
silently training on a mismatched layout.

## Consequences
- Simplest path: one generation pass, no trace-driven replay machinery.
- Changing the observation/action encoding invalidates existing practica; they
  must be regenerated (re-run MCTS). Acceptable: `encode.py` is stable and the
  guard makes the staleness explicit.
- A future decoupled `ReplayPolicy`-based derivation remains possible if encoding
  churn becomes painful.
