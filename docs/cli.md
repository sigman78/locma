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
- `mcts:iters,c,seed,turns` — **cheating** perfect-information MCTS (peeks at the
  opponent's hand); heuristic turn-bounded rollouts by default. Strongest, slow.
- `dmcts:K,I,seed,turns` — **determinized (non-cheating)** MCTS: samples `K`
  opponent hands, runs MCTS (`I` iters) on each, votes. ~as strong as `mcts` but
  fair (public info only). Slower than the heuristics.

`max-guard` and `max-attack` share a "ground" battle: develop the board and
swing at the enemy face, falling back to clearing Guards when the face is not a
legal target. `mcts`/`dmcts` pair their search battle with a greedy draft.

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

## draft-bench
`locma draft-bench [DRAFTS...] [--battle SPEC] [--games N] [--seed S] [--workers W]`
Rank **draft (deck-building) policies in isolation**. Both seats are piloted by the
SAME `--battle` policy and differ only in their draft, so the win-rate edge is pure
deck quality (the draft deals both seats identical offers on a fixed seed; a
self-duel is exactly 0.500). Prints a ranking by average win rate vs the field and
the pair-score matrix. With no `DRAFTS`, ranks all built-in drafts
(`random greedy weighted max-attack max-defense max-guard balanced`).
- A `+rndK` suffix on any draft name (e.g. `balanced+rnd4`) makes exactly `K` of
  its 30 picks uniformly random (a `PartialRandomDraftPolicy` wrapper) — for
  measuring the deck-quality cost of draft noise.
- `--battle` the pilot for both seats: `ground` (default), `greedy`, `scripted`,
  `azlite:100` (strong), `dmcts:K,I` (strong + fair), `ppo:PATH` (deployment net).
- `--games` mirrored game pairs per draft pair (total per pair is `2 × N`).
- `--workers` process-pool workers over the pair grid (0 = all CPUs minus one;
  default 1 = serial). Results are identical to a serial run.

Choose a **strong** pilot — weak heuristics (`ground` vs `greedy`) disagree on the
ranking because each imposes its own style. See `docs/draft-benchmark.md`.

Example: `uv run locma draft-bench --battle azlite:100 --games 60`

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

## train-zoo
`locma train-zoo [--steps-per-opponent N] [--out FILE] [--seed S] [--ent-coef C]`
Train **one** MaskablePPO agent **back-to-back** against a code-declared set of
opponents (a curriculum): the model's weights carry across phases — each phase
swaps the opponent and continues training without resetting. Total budget is
`steps_per_opponent × len(zoo)`. **Requires the `[ml]` extra.**

The opponent set is the `ZOO_OPPONENTS` constant in `locma/envs/training.py`
(currently `greedy → scripted → max-guard → max-attack`); edit that tuple to
change the curriculum. There is no CLI list flag yet — it is deliberately a code
constant for now.

- `--steps-per-opponent` timesteps per opponent phase (default 200000).
- `--out` output path for the saved model (default `model.zip`).
- `--ent-coef` entropy coefficient (default 0.02).
- `--draft-noise K` (also on `train`): make `K` of each deck's 30 draft picks
  uniformly random. The opponent drafts BOTH seats in the battle env, so this
  diversifies the decks the agent trains on without changing the eval draft.

The saved model is a normal PPO artifact — evaluate it like any other policy,
e.g. `locma tournament random scripted greedy max-guard max-attack ppo:zoo.zip
--matrix`. For the rationale and a roadmap of further PPO levers, see
`docs/ppo-review.md`.

Example: `uv run locma train-zoo --steps-per-opponent 200000 --out runs/zoo.zip`

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
