## LoC&M explore kit

This is sandbox to build and explore Legends of Cards & Magic competitive AI, test that ladder, explore
various policies, test them head to head in tournaments, explore hypotesis and dive deep into modern Deep Learning
in the domain of TcG like games.

> **Engine note:** The engine implements single-lane LOCM 1.2 with the 160-card set. Card stats are vendored in `locma/data/cardlist.txt`.

---

## Quickstart

### Install

```bash
# Core install (engine, CLI, policies, eval, tournament)
uv sync

# With ML extras (Gym env + Stable-Baselines3 training)
uv sync --extra ml
```

### CLI commands

```bash
# Play greedy policy against random for 50 mirrored games (seed 0 for reproducibility)
uv run locma play greedy random --games 50 --seed 0

# Evaluate greedy vs random using SPRT — prints verdict (accept_h1 / accept_h0 / continue) and win rate
uv run locma eval greedy --vs random --max-games 200

# Run a round-robin tournament among three policies and print an Elo ratings table
uv run locma tournament random scripted greedy --games 30
```

### Run tests

```bash
uv run pytest
```

### Train an RL agent (requires `[ml]` extra)

```bash
uv run python train.py --steps 50000
```

