# Web panel

`locma serve` hosts a single-page panel for running the kit visually:

```
uv sync --extra server                 # server deps (add --extra ml for model-backed policies)
uv run locma serve                     # http://127.0.0.1:8000
```

In development, run the Vite UI on top (proxies `/api` to :8000):
`cd web && npm install && npm run dev`. In production, `cd web && npm run build`
once — `serve` hosts the bundle at `/`.

Four tabs, hash-routed (`#/experiments`, `#/depot`, `#/replays`, `#/play`);
all stay mounted, so a running job or an in-progress game survives switching.

## Experiments

A visual front-end for the measurement harnesses. Five kinds ship:

- **match** — mirrored A vs B, win rate + Wilson CI + p (`locma play`)
- **noise-floor** — self-play luck baseline + resolution limit
- **league** — round-robin with openskill/Elo table + pair matrix (`locma tournament`)
- **ceiling** — paired per-seed verdict of candidates minus baselines over
  held-out seeds with bootstrap CI (`locma ceiling-eval`, same seed layout)
- **train-zoo** — train a MaskablePPO net against the opponent curriculum
  (`locma train-zoo`; needs `[ml]`, one training job at a time)

Policy fields accept any registry spec, including `depot:` refs
(`vbeam:depot:b0/b0_s0.zip`); an autocomplete list is fed by
`/api/policy-catalog` (baselines + pinned depot models). Configurations can be
saved as **presets** — one JSON file each under `experiments/presets/`,
committed, so presets are shared through git. Three starters ship:
`baseline-league`, `noise-floor-greedy`, `b0-pilot-verdict` (the 10x10 vbeam
vs B0 pilot ruler).

Runs execute as background jobs with live progress and cancel; cells fan out
over a process pool (`--workers`, default all CPUs minus one — same
parallelism as `ceiling-eval --workers`). Finished jobs persist to
`runs/experiments/*.json` (curves included) and reappear in the Runs list
after a restart.

**Live charts.** Expanding a run shows kind-specific visualization, streamed
while it executes (dependency-free inline SVG):

- match/noise-floor: win-rate convergence with the Wilson CI band narrowing
- ceiling: per-seed paired deltas as both arms finish each seed, plus the
  running mean — the verdict visibly converging around the zero line
- league: the pair matrix as a heatmap, filling in pair by pair
- train-zoo: small-multiple training curves (`ep_rew_mean`, value loss,
  entropy, approx-KL, ...) streamed from an SB3 callback

Two streaming channels feed the charts: per-cell results are folded into
series on the collector thread (evals), and long single cells write JSONL
metrics that a tailer thread follows (`training` — progress is timesteps).
Each job also has a log (`view log` in the detail panel): error tracebacks
land there, and training cancel is cooperative — the SB3 callback notices the
cancel file between steps, stops cleanly, and the partial model is still
saved and publishable.

**Training -> depot.** A finished train-zoo run shows the checkpoint path and
a "publish to depot" button, closing the loop train -> watch curves -> verdict
-> promote (`docs/depot.md`).

**Adding a new experiment kind** is server-side only: add a `_Kind` entry in
`locma/server/experiments.py` (param schema + `plan` -> picklable cells +
`reduce`, optionally a `stream` closure for live series). The UI renders the
form from the schema and charts from the series; presets and jobs work
unchanged.

## Depot

The artifact depot (`docs/depot.md`) as a table: versions, provenance
(commit, command, parents, note, meta), pin/local/published status. Actions:
pin a version, pull/push against the configured remote, publish new artifacts
from server-side paths (e.g. `runs/my_s0.zip`), and gc (dry-run first).

## Replays

The replay library and viewer. "Run a matchup" accepts full policy specs
(so planner-vs-baseline replays work), and game-log JSONL files can be
imported row by row.

## Play

Play against any configured policy — the opponent field takes a spec, not
just a baseline name: `vbeam:depot:b0/b0_s0.zip,8,20` plays against the
recipe of record. Model-backed opponents need the `[ml]` extra.
