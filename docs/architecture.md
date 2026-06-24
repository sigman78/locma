# Architecture

## Layering
Pure rules engine at the center; policies, Gym/SB3, stats, and CLI are layers
around it. ML deps stay behind the `[ml]` extra and never enter core.

## Battle rules (LOCM 1.5)

The engine follows the LOCM **1.5** battle rules (runes removed). Verified
against the official referee README, the Strategy-Card-Game-AI `GAME-RULES.md`,
and gym-locm's `Version15BattlePhase`. The subtle, easy-to-miss pieces:

- **Second-player bonus mana ("the coin").** The second player starts with
  `bonus_mana = 1`. Each turn `mana = max_mana + bonus_mana`, so the bonus does
  *not* count toward the 12 max-mana cap (the second player can reach 13). It is
  lost the turn *after* they end a turn having spent all their mana
  (`max_mana > 0 and mana == 0` at turn start). The second player also draws 5
  opening cards to the first player's 4.
- **Damage-based extra draw (no runes).** For every 5 health a player loses to
  *opponent* damage during a round, they draw one extra card at the start of
  their next turn. `PlayerState.damage_counter` accumulates the running
  remainder and is reset each turn; crossing a multiple of 5 bumps `bonus_draw`.
  Only opponent-sourced damage counts — self-damage (a card's own `player_hp`)
  and game damage (deck-out, the 50-turn penalty) do not. All player-health
  changes route through `battle._change_health(p, damage, from_opponent=...)`.
- **Deck-out.** Drawing from an empty deck deals 10 self-damage per missed draw
  (it does not grant a draw). The hand is capped at 8 (overdraw is burned).
- **50-turn penalty.** Once a player has played over 50 turns they take 10
  self-damage at the start of every turn. `gs.turn` counts plies, so the
  threshold is `gs.turn > 100`.

A turn-start deck-out or 50-turn hit can be lethal, so `start_turn` calls
`check_winner` itself (the `Pass` path in `apply_battle` returns before its own
winner check).

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
