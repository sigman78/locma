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
- [Baseline](docs/baseline.md) — reference results for the built-in policies
- [Artifact depot](docs/depot.md) — versioned checkpoints/datasets with provenance (`depot:` refs)

## Install

```bash
uv sync                      # core install
uv sync --extra ml           # + Gym env & SB3 training
uv sync --extra server       # + FastAPI web replay viewer
```

## Quickstart

```bash
uv run locma play greedy random --games 50 --seed 0
uv run locma tournament random scripted greedy --games 30 --matrix
uv run pytest
```

## Web replay viewer

Run matchups and watch replays in the browser — the single-lane field with
both players' hands, draft + battle phases, live card stats and portraits.

```bash
uv sync --extra server                 # one-time: install web server deps
uv run locma fetch-art                 # optional: download card art into the local cache
uv run locma serve                     # start the API on http://127.0.0.1:8000
cd web && npm install && npm run dev   # start the dev UI (proxies /api -> :8000)
```

Then open the Vite URL it prints, run a `greedy vs random` matchup, and scrub
the timeline. **Card art is optional** — without it the viewer draws generated
placeholders. See [`fetch-art`](docs/cli.md) for the art cache.

## License

[MIT](LICENSE) © SiGMan
