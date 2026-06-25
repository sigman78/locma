# 1. Policy spec-string format: positional parameters

Date: 2026-06-25

## Status
Accepted

## Context
Stored game logs carry `policy_a`/`policy_b` as spec strings and `replay`
reconstructs policies via `make_policy(spec)`, hash-checking byte-identical
re-runs. Parametrised policies (MCTS iterations/c/seed, PPO model path) need
their parameters captured so logs stay self-describing and replayable.

## Decision
Spec grammar is `base[:p1,p2,...]`: split name from params on the **first** colon
(`partition(":")`, so paths with colons survive), params split on commas,
positional, trailing params default per preset. Composition of arbitrary
draft×battle halves is a **Python** concern (`Composer`), not part of the string
grammar. The positional order per preset is a frozen contract — reordering breaks
old-log replay.

## Consequences
- Logs are self-describing and replay-safe by construction.
- Positional (vs key=value) is terser but order-sensitive and frozen forever.
- A string composition grammar (e.g. `greedy/mcts`) is a deferred future option.
