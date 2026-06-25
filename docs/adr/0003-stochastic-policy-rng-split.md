# 3. Stochastic policy RNG split

Date: 2026-06-25

## Status
Accepted

## Context
The old `RandomPolicy`/`ScriptedPolicy` each used a single `Random(seed)` consumed
across both phases in sequence (30 draft draws, then battle draws). Splitting into
independent draft/battle halves with their own seeded RNGs changes the random
stream, so old `random`/`scripted` logs no longer replay byte-identically.
Deterministic policies (greedy, max-guard, max-attack, ground) are unaffected.

## Decision
Accept the one-time behaviour change: each half owns an independent `Random(seed)`.
Do not couple the halves with a shared RNG purely for historic byte-compatibility.

## Consequences
- Clean, independent halves; new logs are fully deterministic and replay-safe.
- Pre-refactor `random`/`scripted` replays will fail `--assert-hash`; regenerate
  any stored stochastic baselines.
