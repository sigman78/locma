# Rich Experiment CLI + Repo Hygiene — Design Spec

**Date:** 2026-06-23
**Status:** Approved (brainstorming), pending implementation plan
**Scope:** Grow the LOCM 1.2 explore kit's CLI into a rich experiment surface (play with live render, tournament with pair-score matrix + openskill, noise-floor luck baseline, SPRT, byte-identical replay), built on a new engine trace hook. Plus repo hygiene: docs split, `.editorconfig`, ruff format + lint, and tests.

Builds on the existing kit (see `2026-06-23-locm-12-explore-kit-design.md`). The rules engine, policies, harness, stats (Wilson/binomial/SPRT/Elo), and Gym/SB3 layers already exist and are unchanged except for the trace hook below.

---

## 1. Goals & non-goals

### Goals
- A **trace hook** in the engine so games can be recorded and rendered without slowing the hot path.
- A canonical **game-log format** with a content hash enabling **byte-identical replay**.
- Five experiment **modes** as CLI commands: `play` (+ render/log), `tournament` (+ matrix + openskill), `noise-floor`, `sprt`, `replay`.
- **openskill** ratings alongside the existing Elo.
- Repo hygiene: docs split into `docs/`, `.editorconfig`, ruff format ("prettify") + curated lint, tests for all new behavior.

### Non-goals (for this iteration)
- Graphical replay viewer (rich terminal render only).
- Multi-lane gameplay; engine rules are untouched.
- Parallel match execution (kept sequential and deterministic).
- Prettier/Node toolchain — formatting is Python-only via ruff.

---

## 2. Engine: trace hook (keystone)

`run_game` gains one optional parameter; all other behavior is unchanged:

```python
def run_game(policy0, policy1, seed, cards=None, max_turns=200, on_step=None) -> GameResult
```

- After each `apply_draft_pick` / `apply_battle`, if `on_step` is set, call `on_step(seat, action, gs)`.
- `on_step is None` by default → **zero overhead** for tournament / sprt / noise-floor.
- `GameResult` gains `trace: list | None`, populated only when a recorder is attached.

Two consumers of the hook:
- **Recorder** (`harness/trace.py`): appends `(seat, action)` to a list.
- **Renderer** (`cli/render.py`): prints board state turn-by-turn as the game plays.

**Determinism** (already guaranteed: same seed + policies → same outcome via `policy.reset(seed)`) is the foundation replay relies on.

---

## 3. Game-log format

A logged game is one JSON object per line (a match → a JSONL file):

```json
{"format": 1, "engine_version": "<pkg version>",
 "policy_a": "greedy", "policy_b": "random",
 "seed": 7, "a_seat": 0,
 "actions": [[0, {...}], [1, {...}]],
 "winner": 0, "turns": 23, "hash": "sha256:..."}
```

- `hash = sha256(canonical_json(actions + [winner, turns]))`, where `canonical_json` uses sorted keys and no whitespace.
- Action (de)serialization: `action_to_dict(action)` / `action_from_dict(d)` in `core/actions.py`, canonical and round-trip-stable.
- The header (`policy_a`, `policy_b`, `seed`) carries everything `replay` needs to re-instantiate and re-run.

---

## 4. Commands

Final flat surface (Typer). `eval` is **removed** (its body becomes `sprt`).

| Command | Behavior |
|---|---|
| `play A B [--games N] [--seed S] [--render] [--log FILE]` | Run a mirrored match. Prints win rate + Wilson CI + binomial p (unchanged). `--render`: rich turn-by-turn board view of each game as played. `--log FILE`: write full game-log JSONL (header + actions + hash) per game. |
| `tournament A B C... [--matrix] [--games N] [--seed S] [--reference R]` | Round-robin. Ratings table columns: `policy │ openskill │ elo │ p vs ref`. `--matrix`: render the pair-score matrix as a rich grid. |
| `noise-floor A [--games N] [--seed S]` | Play policy A against an **independent copy of itself**. Prints win rate + Wilson CI, and an explicit **"resolution limit: ±X.XXX"** (CI half-width) — any measured edge smaller than this is indistinguishable from luck. |
| `sprt A --vs B [--p0 .5] [--p1 .6] [--max-games N] [--batch K] [--seed S]` | Sequential probability ratio test (current `eval` body). Stops as soon as evidence decides. Prints verdict (`accept_h1`/`accept_h0`/`continue`) + winrate + CI + n. |
| `replay FILE [--assert-hash] [--render]` | Re-instantiate policies from the log header, re-run from the stored seed, recompute the hash. `--assert-hash`: assert recomputed hash == stored hash; **exit non-zero on mismatch**. `--render`: show the replayed game. |
| `fetch-cards`, `fetch-art` | Unchanged. |

### noise-floor interpretation (documented in `docs/experiments.md`)
- **Stochastic policy** (e.g. random): win rate centers on 0.50; CI width is the measurement floor.
- **Deterministic policy** (greedy/scripted): self-play variance comes only from seat asymmetry + the seed's RNG draws; win rate may sit stably off 0.50. Both readings are documented so the number isn't misread.

---

## 5. New / changed modules

- `core/engine.py` — add `on_step` hook + `GameResult.trace`.
- `core/actions.py` — add `action_to_dict` / `action_from_dict`.
- `harness/trace.py` — recorder, canonical hashing, game-log read/write. (`harness/records.py` match-summary writer stays.)
- `stats/openskill_ratings.py` — `openskill_from_results(pairs) -> {name: (mu, sigma)}` + `ordinal`. `stats/ratings.py` (Elo) kept.
- `cli/render.py` — rich turn-by-turn renderer driven by the `on_step` hook.
- `cli/app.py` — rename `eval`→`sprt`, add `noise-floor` + `replay`, add `--render`/`--log` to `play`, `--matrix` to `tournament`.
- `pyproject.toml` — add `openskill` to `[project.dependencies]`; add `[tool.ruff]` config.

---

## 6. Repo hygiene

- **`.editorconfig`** — UTF-8, LF line endings, final newline, 4-space indent for Python.
- **ruff** (`pyproject.toml`): `line-length = 100`; lint `select = ["E","F","I","UP","B","SIM","PLC"]`; `ruff format` as the formatter ("prettify"). All existing violations fixed in this iteration. Optional `.pre-commit-config.yaml` running `ruff format` + `ruff check`.
- **Docs split** — README becomes a concise intro + links; detail moves to `docs/`:
  - `docs/cli.md` — full command reference (every command, flags, examples).
  - `docs/experiments.md` — methodology: noise-floor as luck baseline, SPRT, ratings (Elo vs openskill), replay determinism.
  - `docs/architecture.md` — engine layering, trace hook, game-log format, determinism guarantee.

---

## 7. Testing

- **Trace round-trip**: record a game, serialize actions, deserialize, assert equality.
- **Hash stability**: same seed + policies → identical hash across runs.
- **Replay**: `replay` of a freshly logged game asserts identical hash; a tampered log fails (non-zero exit).
- **Action (de)serialization**: every action variant round-trips.
- **openskill**: `openskill_from_results` orders a dominant policy above a weak one; ordinal monotonic.
- **noise-floor**: a stochastic policy vs itself yields win rate within CI of 0.50.
- **CLI smoke tests**: each command runs end-to-end on a tiny game count and exits 0.

---

## 8. Decomposition

Single spec. The hygiene track (section 6) is small and touches the same files as the CLI work, so it ships together. One implementation plan.
