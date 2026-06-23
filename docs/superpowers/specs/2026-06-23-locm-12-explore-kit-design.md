# LoC&M 1.2 Explore Kit — Design Spec

**Date:** 2026-06-23
**Status:** Approved (brainstorming), pending implementation plan
**Scope:** A Python kit to simulate single-lane Legends of Code & Magic (LOCM) 1.2, plug in policies, run head-to-head matches and tournaments, and statistically evaluate hypotheses (notably "is policy X better than random?"). Built so Gymnasium + Stable-Baselines3 (SB3) sit on top for training/eval without leaking into the core.

---

## 1. Goals & non-goals

### Goals
- Clean-room, dependency-light **single-lane LOCM 1.2** rules engine (160 cards, draft + battle).
- Pluggable **Policy** interface with both draft and battle decision hooks.
- **Harness** to run reproducible matches and tournaments, with parallelism and on-disk result logging.
- **Statistics** layer: Wilson confidence intervals, binomial test, SPRT sequential testing, Elo/TrueSkill ratings.
- **Gymnasium env + MaskablePPO** adapter so SB3 can train agents that drop straight into the harness.
- **Typer CLI** as the single entry point for all of the above.
- Vendored, verifiable card stat data; best-effort card art download.

### Non-goals (for v1)
- Multi-lane (1.5) gameplay. Single lane only.
- Card artwork as a first-class, license-audited pipeline (best-effort only).
- A graphical UI / replay viewer (the JSONL logs are designed to enable one later).
- Distributed/cluster training.

---

## 2. Architecture & layering

Guiding principle: a **pure rules engine at the center**; policies, Gym, SB3, CLI, and stats are layers around it. ML dependencies live behind an optional extra and never enter the core.

```
┌─────────────────────────────────────────────────────────┐
│ CLI (Typer)  →  play · tournament · eval · train · fetch  │
├─────────────────────────────────────────────────────────┤
│ Harness:  match runner · tournament · stats (CI/SPRT/Elo) │
├──────────────┬──────────────────────┬─────────────────────┤
│ Policies     │  Gym adapter         │  Data layer          │
│ (Random,     │  (Gymnasium env +    │  (card DB loader,    │
│  Greedy,     │   action masking →   │   art fetcher,       │
│  Scripted,   │   wraps engine)      │   manifest)          │
│  SB3 wrapper)│  → SB3 trains here   │                      │
├──────────────┴──────────────────────┴─────────────────────┤
│            CORE ENGINE  (pure Python, no ML deps)           │
│  GameState · draft · battle rules · legal-action gen ·      │
│  card model · seeded RNG · deterministic & replayable       │
└─────────────────────────────────────────────────────────┘
```

### Package layout
```
locma/
  core/        engine: state, cards, rules, actions, rng
  data/        cardlist loader, art fetcher; assets/ + cardlist.txt (vendored)
  policies/    base Policy, random, greedy, scripted, sb3_wrapper
  envs/        gymnasium env + action encoding/masking
  harness/     match runner, tournament, parallel, result logging (JSONL)
  stats/       wilson CI, binomial test, SPRT, elo/trueskill
  cli/         Typer app
tests/         engine rule tests, golden-game tests, stats tests
docs/superpowers/specs/
pyproject.toml  ([ml] extra = gymnasium, sb3-contrib, stable-baselines3, torch)
```

### Stack
- Python 3.11+
- CLI: **Typer** + **Rich** (tables/output)
- Numerics/stats: **numpy**, **scipy**
- ML (optional `[ml]` extra): **Gymnasium**, **stable-baselines3**, **sb3-contrib** (MaskablePPO), **torch**; optional **trueskill**
- Tests: **pytest**
- Dependency manager: **uv** (plain pip/venv as fallback) — Windows-friendly

---

## 3. Core engine model

### Card model
- `Card` — immutable card definition loaded from `cardlist.txt`: `id, name, type, cost, attack, defense, abilities, player_hp, enemy_hp, card_draw`.
- `CardInstance` — wraps a `Card` with mutable per-game state: instance id, current attack/defense, can-attack flag, possibly modified abilities.
- Types: `CREATURE`, `GREEN_ITEM` (buff own creature), `RED_ITEM` (debuff enemy creature), `BLUE_ITEM` (damage/effect, may target face).
- Abilities as a flag set: `Breakthrough, Charge, Drain, Guard, Lethal, Ward`.

### GameState
- Per player: hp, mana, max-mana, rune/next-rune state, deck, hand, board (≤6, single lane).
- Global: whose turn, turn count, phase, seeded RNG.
- **Cheaply copyable** (for future search/lookahead policies) and **fully reconstructable** from `(seed, action_log)` (replayable).

### Phases
- **DraftState:** 30 rounds; each presents 3 cards; `legal_actions` = pick index `{0,1,2}`. Both players draft from the same seeded, mirrored triplets per 1.2 rules.
- **BattleState:** mana ramps 1→max per turn. Legal actions: `SUMMON`, `ATTACK` (creature→creature or creature→face, Guard-respecting), `USE` (item with target rules), `PASS`.
  - Rules: Guard forces targeting; Ward negates next instance of damage; Lethal kills any creature it damages; Drain heals controller by damage dealt; Breakthrough tramples excess damage to face; Charge allows attacking the turn summoned; rune/health thresholds trigger bonus card draw.
  - Win: opponent hp ≤ 0 (plus deck-out/rune rules).

### Action model
- Engine contract: `legal_actions(state) -> list[Action]` and `apply(state, action) -> state`. Both policies and the Gym env consume the same `legal_actions`.

### Determinism
- All randomness (shuffles, draft triplets, draws) flows through a single seeded RNG. Same seed + same policies ⇒ identical game. Foundation for trustworthy stats and reproducible bugs.

### Correctness strategy
- Unit tests per rule (Guard targeting, Ward, Lethal, Breakthrough math, Drain, mana/rune draw).
- "Golden game" tests (fixed seed + scripted actions → known outcome).
- Where rules are ambiguous, cross-check behavior against gym-locm / the reference referee.

---

## 4. Policy interface

```python
class Policy(Protocol):
    name: str

    def draft_action(self, view: DraftView, legal: list[int]) -> int:
        """Pick 0/1/2 from the offered triplet."""

    def battle_action(self, view: BattleView, legal: list[Action]) -> Action:
        """Choose one legal battle action (may be PASS)."""

    def reset(self, seed: int | None = None) -> None:
        """Optional: clear per-game internal state."""
```

- **Two independent hooks** enable mix-and-match: `CompositePolicy(draft=RandomDraft(), battle=GreedyBattle())` — directly supports hypotheses like "random draft + smart battle".
- **`view`** is a sanitized, read-only snapshot (opponent hand/deck masked), not the live `GameState`. Prevents cheating/mutation and is the basis for Gym observations.
- **`legal`** is always supplied. The harness validates the returned action; an illegal/None return is a **hard error** (fail fast, never silently PASS).
- Policies may be stateful (SB3 net, search cache); `reset()` marks game boundaries.

### v1 baselines
- `RandomPolicy` — uniform over `legal` for both hooks, seeded from the game RNG.
- `GreedyPolicy` — curve/stat heuristic draft + greedy-trade battle heuristic (Guard-aware, uses Lethal/Ward, attacks face when ahead).
- `ScriptedPolicy` — degenerate lower bound (cheapest legal / pass).
- `SB3Policy` — wraps a loaded SB3 model; `view`→observation, apply mask, net output→`Action`.

---

## 5. Evaluation & tournament layer

### Match runner
- `run_match(policy_a, policy_b, games=N, seed=...) -> MatchResult` (wins/losses/draws + per-game records).
- **Side balancing:** each logical matchup played as *mirrored seed pairs* (same seed, A-first then B-first) to cancel LOCM's first-player advantage.
- **Reproducibility & logging:** each game recorded as `(seed, side, action_log, winner)`, streamed to **JSONL** on disk for re-analysis, audit, and replay.

### Parallelism
- Games within a match and matchups within a tournament fan out via `concurrent.futures`/`multiprocessing`. Pure, seed-driven engine ⇒ deterministic regardless of worker count.

### Stats module (pure functions over win counts)
- `wilson_ci(wins, n)` → 95% interval on win rate.
- `binomial_test(wins, n, p0=0.5)` → p-value vs baseline/random.
- `sprt(...)` → sequential test with configurable H0/H1 win-rate hypotheses + α/β bounds; early-stop, capped at max-N.
- Ratings: `elo` (self-contained) and optional `trueskill` (via `trueskill` package).

### Tournament
- `run_tournament(policies, ...)` → round-robin (all pairs, mirrored) producing:
  - win-rate matrix (every pair),
  - leaderboard with Elo/TrueSkill + CI,
  - per-matchup p-values vs a designated reference (e.g. Random).

### "Vs random" hypothesis flow
- Pick policy X + `RandomPolicy`, run with SPRT (H0: winrate ≤ 0.5, H1: ≥ δ). Harness reports verdict + CI + games-used. Rich tables in terminal; JSONL for the record.

---

## 6. Gym/SB3 adapter

Challenge: variable, illegal-heavy action spaces. The engine's `legal_actions` already exists, so the adapter is just encode ↔ decode + masking.

- **`LOCMEnv(gymnasium.Env)`** wraps a single seat; opponent is a fixed `Policy` passed in (self-play = pass a snapshot of the learning policy). Standard single-agent env.
- **Observation:** fixed-length `float32` vector from the sanitized `view` — player/opponent hp, mana, rune state, hand-card features, board-slot features (≤6 per side), turn. Shares encoders with the policy `view`.
- **Action space:** fixed discrete space covering the max action set (summon slot × hand index, attack attacker×target incl. face, use-item × target, pass). Illegal actions handled via **action mask** (`action_masks()`), trained with **MaskablePPO** (sb3-contrib). Mask derived from `legal_actions`.
- **Env flavors:** `BattleEnv` (draft by a fixed policy, agent learns battle) now; draft-capable variant later. Supports studying each phase in isolation.
- **Reward:** win/loss terminal reward by default; optional shaped-reward hook (hp differential).
- **`SB3Policy`** is the inverse path: load model → encode `view`, mask, pick, decode to engine `Action`. Trained models plug into tournaments alongside Random/Greedy.
- A tiny `train.py` (MaskablePPO, minimal) ships as a reference example.

---

## 7. Data layer (cards + best-effort art)

- **`cardlist.txt` (160 cards):** fetched from the canonical LOCM source (official `acatai/LegendsOfCodeAndMagic` repo / gym-locm mirror) by a `fetch-cards` command; parsed by `load_cards()` into the `Card` table. **Vendored in-repo** (tiny, permissively licensed) so the engine works offline; the fetch command refreshes/verifies. A checksum + count assertion (`== 160`) guards bad downloads. Exact source URLs to be confirmed and recorded during implementation.
- **Best-effort art:** a separate `fetch-art` command pulls whatever card images exist (CodinGame/community mirrors) into `locma/data/assets/`, writing `manifest.json` (card id → file, source URL, license note). **Optional and non-blocking** — failures log a warning and skip; the core never imports it.
- Availability of both will be verified during implementation; if art is unavailable, stats-only stands and the command degrades gracefully.

---

## 8. CLI surface (Typer)

- `locma play A B [--games N] [--seed S]` — head-to-head match, win rate + Wilson CI + p-value.
- `locma tournament P1 P2 ... [--games N]` — round-robin, win-rate matrix + leaderboard (Elo/TrueSkill).
- `locma eval X [--vs random] [--sprt] [--delta δ]` — hypothesis test with SPRT early-stop.
- `locma train [--steps N] [--out model.zip]` — MaskablePPO reference training.
- `locma fetch-cards` / `locma fetch-art` — data acquisition/refresh.

All results render as Rich tables and stream to JSONL.

---

## 9. Milestones (rough sequencing)

1. **Data + card model:** vendor/parse `cardlist.txt`, `Card`/`CardInstance`, `load_cards()` + count test.
2. **Engine core:** `GameState`, draft phase, battle phase rules, `legal_actions`/`apply`, seeded RNG; rule unit tests + golden game.
3. **Policies:** `Policy` protocol, `CompositePolicy`, Random + Scripted + Greedy.
4. **Harness + stats:** match runner (mirrored), parallelism, JSONL logging; Wilson/binomial/SPRT/Elo.
5. **CLI:** `play`, `tournament`, `eval`, `fetch-cards`/`fetch-art`.
6. **Gym/SB3:** `LOCMEnv` + masking, `SB3Policy`, `train.py`; `[ml]` extra.
7. **Best-effort art** fetch (can slot in any time after step 1).

---

## 10. Open items to resolve during implementation
- Confirm exact canonical URLs/format for `cardlist.txt` and pin a vendored copy.
- Confirm whether usable card art exists and where; otherwise degrade `fetch-art` gracefully.
- Finalize the fixed action-space sizing and observation-vector schema (documented when engine state is concrete).
