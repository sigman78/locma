## LoC&M explore kit

This is sandbox to build and explore Legends of Cards & Magic competitive AI, test that ladder, explore
various policies, test them head to head in tournaments, explore hypotesis and dive deep into modern Deep Learning
in the domain of TcG like games.

> **Engine note:** The engine implements single-lane LOCM 1.2 with the 160-card set. Card stats are vendored in `locma/data/cardlist.txt`.

---

## Documentation

- [CLI reference](docs/cli.md) — every command and flag
- [Experiment methodology](docs/experiments.md) — noise floor, SPRT, ratings, replay
- [Architecture](docs/architecture.md) — engine, trace hook, game-log format

## Install

```bash
uv sync                      # core install
uv sync --extra ml           # + Gym env & SB3 training
```

## Quickstart

```bash
uv run locma play greedy random --games 50 --seed 0
uv run locma tournament random scripted greedy --games 30 --matrix
uv run pytest
```
