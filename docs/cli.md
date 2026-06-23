# CLI Reference

All commands run via `uv run locma <command>`.

## play
`locma play A B [--games N] [--seed S] [--render] [--log FILE]`
Run a mirrored match. Prints win rate + 95% Wilson CI + binomial p-value.
- `--render` renders each game turn-by-turn as played.
- `--log FILE` writes a game-log JSONL (one record per game) enabling `replay`.

Example: `uv run locma play greedy random --games 50 --seed 0 --log run.jsonl`

## tournament
`locma tournament A B C... [--games N] [--seed S] [--reference R] [--matrix]`
Round-robin. Ratings table: policy | openskill (ordinal) | elo | p vs reference.
- `--matrix` prints the pair-score matrix (row win rate vs column).

Example: `uv run locma tournament random scripted greedy --games 30 --matrix`

## noise-floor
`locma noise-floor A [--games N] [--seed S]`
Plays A against an independent copy of itself — the luck baseline. Prints win
rate, CI, and the resolution limit (CI half-width).

Example: `uv run locma noise-floor greedy --games 500`

## sprt
`locma sprt X --vs B [--p0 0.5] [--p1 0.6] [--max-games N] [--batch K] [--seed S]`
Sequential probability ratio test; stops as soon as evidence decides. Prints
verdict (accept_h1 / accept_h0 / continue), win rate, CI, and n.

Example: `uv run locma sprt greedy --vs random --max-games 200`

## replay
`locma replay FILE [--assert-hash] [--render]`
Re-simulates each logged game from its seed and recomputes the content hash.
- `--assert-hash` exits non-zero on any mismatch.
- `--render` shows each replayed game.

Example: `uv run locma replay run.jsonl --assert-hash`

## fetch-cards / fetch-art
Refresh the vendored card list / best-effort art download.
