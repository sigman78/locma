# League Self-Play — handoff pointer

This carries the **design + plan only** — no code yet. It's authored on a CPU box;
the ~3–4 hr training is the reason to move it to a **faster PC** (RTX 4080 box or
M1 Max) in a fresh Claude Code session.

Branch: **`feat/ppo-autoreg-action`** (the reactive-PPO exploration branch; the
AR-head study already landed here — verdict: no-help).

## What's here

- `docs/ppo-league-selfplay-design.md` — the approved spec (rationale, locked decisions).
- `docs/ppo-league-selfplay-plan.md` — the bite-sized TDD plan (Tasks 1–6 build the
  tooling; an Execution Runbook R0–R4 runs the study).
- `docs/ppo-league-selfplay-HANDOFF.md` — this file.

## Pick-up procedure on the faster PC

1. `git fetch && git checkout feat/ppo-autoreg-action && git pull`
2. `uv sync --extra ml --extra dev`
3. Verify the device: `uv run --extra ml python -c "import torch; print('cuda', torch.cuda.is_available())"`
   → `True` uses the GPU; else it runs on CPU/MPS (the token net is small — CPU is
   fine, GPU helps the attention modestly).
4. Start a fresh Claude session and execute `docs/ppo-league-selfplay-plan.md` with
   **superpowers:subagent-driven-development** (recommended) or **executing-plans**.
5. CI gate before every commit: `uv run --extra dev ruff check . && uv run --extra dev
   ruff format --check . && uv run --extra ml --extra dev pytest -q`.

## Opening prompt for the fresh session

After the pick-up procedure above, open `claude` in this repo and paste this verbatim:

```
I'm continuing the "League Self-Play" study. The design and a bite-sized TDD
implementation plan are already committed on this branch (feat/ppo-autoreg-action).

Read these first, in order:
  docs/ppo-league-selfplay-HANDOFF.md
  docs/ppo-league-selfplay-design.md
  docs/ppo-league-selfplay-plan.md

Then execute the plan task-by-task using the superpowers:subagent-driven-development
skill. Build and verify Tasks 1-6 (the tooling) first; the tests are CPU-fast.

Once the tooling is green, run the Execution Runbook R0-R4 (this is the long part,
~3-4 hr): R0 trains the 800k token base, R1 runs the 6-round FSP league (watch
runs/league/league.csv), R2 is the paired verdict of the best league net vs the
base, R3 plugs the best net in as a netdmcts oracle (avg-hard3 vs 0.817), R4 writes
the result into docs/worklog.md. Run the long steps in the background.

Run everything via `uv run` with --extra ml (training/eval) and --extra dev
(tests/lint). Before each commit, pass the CI gate:
  uv run --extra dev ruff check .
  uv run --extra dev ruff format --check .
  uv run --extra ml --extra dev pytest -q
Commit per task with the messages the plan specifies. Start with Task 1.
```

To run inline instead of dispatching subagents, swap the skill line for
"execute with superpowers:executing-plans".

## The one-line goal

Turn the throwaway 2-snapshot self-play into a proper, tracked **fictitious
self-play league** (all past snapshots + baselines) for the token net — see whether
it clears the prior reactive plateau **0.639** and/or yields a **netdmcts oracle**
stronger than the current **0.817**.

## Key facts the fresh session should not re-derive

- **Token net only** (flat self-play is a known regression). Base = 800k token zoo.
- **Opponent is built as a Python object, n_envs=1** — snapshot paths can't ride a
  flat opponent spec string (the plumbing constraint the whole design works around).
- **`target_kl=0.025`** keeps self-play updates stable (the prior probe relied on it).
- Prior results to beat: reactive plateau **0.639** (selfplay-r2), search **0.817**
  (netdmcts with that same net as oracle).
