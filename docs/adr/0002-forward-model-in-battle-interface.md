# 2. Forward model in the battle interface

Date: 2026-06-25

## Status
Accepted

## Context
`BattleView` is imperfect-information (opponent hand contents never exposed), so a
search policy like MCTS cannot build a forward model from the view alone. It needs
the full `GameState` to clone and simulate.

## Decision
Add an optional `state` argument to `battle_action(view, legal, state=None)`. The
engine (`run_game`) and interactive harness pass the live `GameState`; search
policies deep-copy and drive `battle_legal`/`apply_battle`. All other battle
policies ignore it. This is cheating perfect-information MCTS — a standard strong
baseline. `BattleEnv` does not pass it yet (training-MCTS-opponent deferred).

## Consequences
- MCTS is cheap to build and deterministic (replay-safe).
- The mutable engine state leaks into the battle-policy contract (mild
  architecture-purity cost) and the AI "cheats."
- A `Simulator` wrapper (`clone()/legal()/apply()/winner()`) is the noted future
  refinement once a second search policy wants it; determinized/IS-MCTS is the
  honest-imperfect-information upgrade.
