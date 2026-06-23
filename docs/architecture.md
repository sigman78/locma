# Architecture

## Layering
Pure rules engine at the center; policies, Gym/SB3, stats, and CLI are layers
around it. ML deps stay behind the `[ml]` extra and never enter core.

## Trace hook
`run_game(..., on_step=None)` calls `on_step(seat, action, gs)` after each
applied policy decision (draft pick int, or battle Action). Default `None` means
zero overhead for tournaments/sprt/noise-floor. Two consumers:
- `harness/trace.Recorder` collects `(seat, action)` pairs.
- `cli/render.GameRenderer` prints turn-by-turn.

## Game-log format
One JSON object per line (a match -> a JSONL file):
`{format, engine_version, policy_a, policy_b, seed, a_seat, actions, winner,
turns, hash}`. `actions` is the serialized trace; `hash` is
`sha256:<hexdigest>` of `canonical_json(actions + [winner, turns])`.

## Determinism guarantee
`policy.reset(seed)` + `random.Random(seed)` make each game reproducible
regardless of play order, which is what makes byte-identical replay possible.
