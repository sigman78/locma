# Ideas / future refactors

Speculative directions, captured so they aren't lost. Nothing here is committed
work — each entry states the problem it solves, the change, the cost, and when it
would be worth doing.

## Background: the snapshot-ordering problem

`GameState` is mutated in place, and the replay recorder reads it around each
action. A turn-ending `Pass` fuses two things inside `apply_battle`: the action
itself and the turn transition (`end_turn` = flip `gs.current`, bump `gs.turn`,
`start_turn` draws for the opponent). So "the state after a `Pass`" is the
*opponent's* next turn, not the actor's — the actor's end-of-turn state exists for
zero instructions before it is overwritten.

The shipped fix is **decision-point recording**: record each step from the state
*before* the action is applied (`on_pre_step`), which is uniformly in the acting
seat's perspective (`state.current == seat`). This was chosen over moving
`end_turn` out of `apply_battle`, because `apply_battle(Pass) == end_turn` is a
load-bearing contract that `envs/battle_env.py` (the RL loop) and direct-apply
tests depend on without going through `run_game`. The ideas below are the larger
hammers that would attack the same root causes more structurally.

## 1. Immutable `GameState` transitions

**Smell it fixes.** Observe-after-mutate. `snapshot()` exists *only* because `gs`
is mutable and aliased — every recorded state must be deep-copied or it would be
clobbered by the next mutation. That deep copy is pure ceremony forced by the
mutation model, and the same aliasing invites a class of bugs (a held reference
silently changing under you).

**The change.** Make state transitions return a new value instead of mutating in
place: `apply_battle(gs, action) -> GameState` (and likewise `end_turn`,
`start_turn`). Recording becomes "keep the value you were handed" — no `snapshot()`,
no defensive copying, no hook-timing subtlety, because each state *is* an
immutable value pinned in time.

**Cost.** Invasive. Every mutator in `core/battle.py` and every caller
(`engine.run_game`, `envs/battle_env.py`, the whole test suite that pokes `gs`
in place) has to thread the returned state. Likely a measurable allocation cost
in the hot tournament/SPRT loop unless paired with structural sharing
(persistent data structures), which is more machinery still.

**When worth it.** If aliasing bugs recur, or if we want trivially-correct
time-travel / branching (explore "what if P0 had attacked instead?") — immutable
states make divergent futures free. **Note it does not fix ordering by itself:**
an immutable `apply_battle(Pass)` that still fuses `end_turn` returns the
opponent's turn. Combine with idea #2 or with decision-point recording.

## 2. Event / delta stream from the engine

**Smell it fixes.** Conflation. `Pass` secretly bundles the turn transition, so a
single "action applied" notification hides three distinct facts (turn ended,
seat flipped, opponent drew). Decision-point recording dodges this; it does not
*decompose* it.

**The change.** Have the engine emit a stream of atomic events instead of (or
alongside) the in-place mutation:

```
ActionApplied(seat, action)      # the pure action effect, no turn flip
TurnEnded(seat)
TurnStarted(seat, draws=[...])   # the opponent's start-of-turn draw, owned by them
DamageDealt(target, amount) / UnitDied(iid) / ...   # optional finer grain
```

A `Pass` emits `ActionApplied(Pass) -> TurnEnded(0) -> TurnStarted(1, draw=…)`.
The recorder (and the web viewer) fold the events; the ordering problem evaporates
because the turn transition is its own event, not a hidden rider on a snapshot.

**What it unlocks.**
- **Animated replays.** The web `fx` layer currently *diffs consecutive snapshots*
  to reconstruct what happened (`web/src/lib/fx.ts`). With explicit events it
  renders the truth directly — animate the opponent's draw separately from the
  pass, show each point of damage, sequence a multi-effect turn.
- **Undo / branching** and **netcode-style deltas** (ship events, not whole boards)
  become natural.
- The stored replay shrinks to `opening + events`; per-step snapshots become a
  derived cache, not the source of truth.

**Cost.** The most machinery of the three: an event vocabulary, emit points
threaded through `core/battle.py`, fold logic in every consumer, and a migration
for stored replays. Risk of over-engineering if all we ever need is "show the
board after each move."

**When worth it.** When replay *presentation* gets ambitious (animation, scrubbing
mid-effect, diff-based sync) or when we want the engine to be replayable/undoable
as a first-class capability. Until then, decision-point snapshots are the cheaper
90%.

## 3. Don't store snapshots at all — derive them

**Observation.** The game is deterministic from `seed + actions`
(see *Determinism guarantee* in `architecture.md`), so `opening + actions` is a
*complete* replay. The per-step `state` snapshots are a convenience cache for
consumers that don't run the engine (the web client today).

**The change.** Treat snapshots as a derived view. The recorder — or any consumer
with an engine — replays the action trace through its own stepping function and
snapshots at exactly the observation points it wants (e.g. always pre-action),
fully decoupling snapshot timing from `run_game`'s hooks. `run_game` then only has
to produce the action trace.

**Cost / caveat.** Needs an engine on the consumer side. The web client is
TypeScript and has no engine, so it still needs materialized snapshots shipped to
it — unless we compile the rules to WASM or port a thin replayer. Reasonable the
day the web gets an engine; overkill while snapshots must travel pre-rendered.

**When worth it.** If a Python/WASM engine lands on the web side, or if storage
size of snapshots ever becomes a concern — drop them and derive.
