# PPO Ceiling Study — handoff pointer

This branch (`feat/ppo-ceiling-study`) carries the **design + plan only** — no code yet.
It was authored on a machine without CUDA; execution is meant to happen on the
**RTX 4080 (16 GB) box** (or M1 Max via MPS) in a fresh Claude Code session.

## What's here

- `docs/ppo-ceiling-study-design.md` — the approved spec (rationale, locked decisions).
- `docs/ppo-ceiling-study-plan.md` — the bite-sized TDD implementation plan (Tasks 1–11
  build the tooling; an Execution Runbook R0–R6 runs the study).
- `docs/ppo-ceiling-study-HANDOFF.md` — this file.

## Pick-up procedure on the GPU box

1. `git fetch && git checkout feat/ppo-ceiling-study`
2. `uv sync --extra ml --extra dev --extra sweep`  *(the `sweep` extra — optuna +
   tensorboard — is added by Task 2; if it's not in `pyproject.toml` yet, run
   `uv sync --extra ml --extra dev` and let Task 2 add it.)*
3. Verify CUDA: `uv run --extra ml python -c "import torch; print(torch.cuda.is_available())"`
   → `True` (else fall back to `--device mps` or `cpu`).
4. Start a fresh Claude session and execute `docs/ppo-ceiling-study-plan.md` with
   **superpowers:subagent-driven-development** (recommended) or **executing-plans**.
5. CI gate before every commit: `uv run --extra dev ruff check . && uv run --extra dev
   ruff format --check . && uv run --extra dev pytest -q`.

## The one-line goal

Settle, with a symmetric +0.03 paired-difference verdict, whether the token PPO net's
~0.60 avg-hard3 plateau is the true reactive ceiling or an under-tuning artifact —
with live telemetry, an Optuna sweep, and obs-encoding variants, all on sb3.
