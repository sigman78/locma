# CLI Reference

All commands run via `uv run locma <command>`.

## Policies

Any command that takes a policy name (`play`, `tournament`, `noise-floor`,
`sprt`, `--opponent`) accepts one of:

- `random` — uniform random legal action.
- `scripted` — random draft + fixed aggressive battle script (green items →
  attack face/Guard → summon → remaining items), targets chosen at random.
- `greedy` — stat-based draft + greedy battle (lethal/trade heuristic).
- `max-guard` — draft prefers Guard creatures; aggressive ground battle.
- `max-attack` — draft prefers highest-attack creatures; aggressive ground battle.

`max-guard` and `max-attack` share a "ground" battle: develop the board and
swing at the enemy face, falling back to clearing Guards when the face is not a
legal target.

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

## train
`locma train [--steps N] [--out FILE] [--opponent P] [--seed S]`
Train a MaskablePPO agent on the battle env against a fixed opponent policy,
then save the model. **Requires the `[ml]` extra** (`uv sync --extra ml`); without
it the command exits with a clear error.
- `--steps` total training timesteps (default 50000).
- `--out` output path for the saved model (default `model.zip`).
- `--opponent` opponent policy (see Policies above; default random).

Example: `uv run locma train --steps 50000 --opponent random --out model.zip`

## fetch-cards
`locma fetch-cards`
Refresh the vendored card list from the upstream source.

## fetch-art
`locma fetch-art [--force]`

Opt-in download of card portrait art into the local cache
(`locma/data/assets/`, gitignored). Skips already-cached files unless `--force`.
Portraits are sourced from `legendsofcodeandmagic.com`. The web replay viewer
(`locma serve`) reads this cache and falls back to generated placeholders for
any card whose art is missing, so fetching is optional.

- `--force` re-download even if a file is already cached.

Example: `uv run locma fetch-art`

> Card art is downloaded for local use only; seek permission from the authors
> before redistribution.

## serve
`locma serve [--host 127.0.0.1] [--port 8000] [--replay-dir replays] [--asset-dir locma/data/assets] [--gamelog-dir .]`

Start the local web replay viewer API (run matchups, browse/import replays, and
play them back in the browser). **Requires the `[server]` extra**
(`uv sync --extra server`); without it the command exits with a clear error.

- `--host` / `--port` bind address (default `127.0.0.1:8000`).
- `--replay-dir` where generated replays are persisted (gitignored, default `replays/`).
- `--asset-dir` card-art cache served at `/api/art/{id}` (default `locma/data/assets`; see `fetch-art`).
- `--gamelog-dir` directory scanned for `*.jsonl` game-logs to import (default `.`).

In development, also run the Vite UI which proxies `/api` to this server:
`cd web && npm install && npm run dev`. In production, build the bundle
(`cd web && npm run build`) and `serve` will host it at `/`.

Example: `uv run locma serve --port 8000`
