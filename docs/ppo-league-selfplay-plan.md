# League Self-Play (Token PPO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tracked fictitious-self-play (FSP) *league* loop for the token PPO net — each round trains against a per-episode mix of all past frozen snapshots + baselines — plus a generic avg-hard3 eval so the best league net can be tested as a netdmcts oracle.

**Architecture:** A new `locma/envs/league.py` orchestrates rounds: build a `MixedOpponentPolicy` from `[ppo:<snap> …] + baselines` as a **Python object** (n_envs=1 `DummyVecEnv` — snapshot paths can't ride a flat opponent spec string), continue the same token model with `set_env` + `learn(reset_num_timesteps=False)` under a conservative `target_kl=0.025`, snapshot into the pool, eval avg-hard3 → CSV. Eval reuses/extends `ar_study`.

**Tech Stack:** Python, PyTorch, stable-baselines3 2.9 + sb3-contrib 2.9 (`MaskablePPO`), gymnasium, numpy, typer, pytest, ruff, uv.

## Global Constraints

- **Net = token** (`obs_mode="token"`) — flat self-play is a known regression.
- **FSP league:** opponent pool each round = `[ppo:<snap> for every past snapshot] + [scripted, max-guard, max-attack]`, uniform per-episode sampling by `MixedOpponentPolicy`.
- **n_envs=1 `DummyVecEnv`, opponent built as a Python object** (not a spec string).
- **Continue the same model across rounds** via `set_env` + `learn(..., reset_num_timesteps=False)`; **`target_kl=0.025`**.
- **6 rounds × 200k** on an **800k token zoo base** (round 0) — study parameters; the tooling must accept them as arguments (not hardcode).
- **Metric:** `avg-hard3` = mean win-rate vs {scripted, max-guard, max-attack}, deterministic, held-out seeds (`1_000_000+`).
- **No new dependencies.** Everything via `uv run` (`--extra ml` for ML, `--extra dev` for tests/lint).
- **CI gate before every commit:** `uv run --extra dev ruff check . && uv run --extra dev ruff format --check . && uv run --extra ml --extra dev pytest -q`.
- **Additive:** touch only `league.py` (new), `ar_study.py` (add one function + DRY delegation), `app.py` (two commands). No existing behavior changes.

## File Structure

| File | Responsibility | ML dep |
|------|----------------|--------|
| `locma/envs/league.py` (new) | `league_pool_specs`, `write_league_csv`, `build_league_opponent`, `_league_env`, `run_league` | yes (lazy) |
| `locma/harness/ar_study.py` (modify) | add `avg_hard3_spec(spec, …)`; make `hard3_per_seed` delegate to it | no (numpy) |
| `locma/cli/app.py` (modify) | `selfplay-league` + `hard3-eval` commands | no |
| `tests/test_league.py` (new) | pool specs, CSV, opponent/env builders, smoke league | mixed |
| `tests/test_ar_study.py` (modify) | test `avg_hard3_spec` | no |

---

### Task 1: Pool specs + CSV writer (torch-free)

**Files:**
- Create: `locma/envs/league.py`
- Test: `tests/test_league.py`

**Interfaces:**
- Produces:
  - `DEFAULT_BASELINES: tuple[str, ...] = ("scripted", "max-guard", "max-attack")`
  - `league_pool_specs(snapshots: list[str], baselines=DEFAULT_BASELINES) -> list[str]`
  - `write_league_csv(path, rows: list[dict]) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_league.py
import csv

from locma.envs.league import DEFAULT_BASELINES, league_pool_specs, write_league_csv


def test_pool_specs_snapshots_then_baselines():
    specs = league_pool_specs(["a.zip", "b.zip"], ["scripted"])
    assert specs == ["ppo:a.zip", "ppo:b.zip", "scripted"]


def test_pool_specs_default_baselines_and_length():
    specs = league_pool_specs(["a.zip"])
    assert specs == ["ppo:a.zip", *DEFAULT_BASELINES]
    assert len(specs) == 1 + len(DEFAULT_BASELINES)


def test_write_league_csv_roundtrip(tmp_path):
    rows = [
        {"round": 0, "snapshot": "round0.zip", "avg_hard3": 0.601, "n_seeds": 150},
        {"round": 1, "snapshot": "round1.zip", "avg_hard3": 0.632, "n_seeds": 150},
    ]
    p = tmp_path / "sub" / "league.csv"
    write_league_csv(p, rows)
    with open(p, newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert [r["round"] for r in got] == ["0", "1"]
    assert got[1]["snapshot"] == "round1.zip"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_league.py -q`
Expected: FAIL — `ModuleNotFoundError: locma.envs.league`

- [ ] **Step 3: Write minimal implementation**

```python
# locma/envs/league.py
"""League (fictitious self-play) loop for the token PPO net.

Each round trains the current net against a per-episode mix of all past frozen
snapshots + the ground baselines, then snapshots itself into the pool. See
docs/ppo-league-selfplay-design.md. ML imports are lazy so the pure helpers
(pool specs, CSV) stay import-safe without the [ml] extra."""

from __future__ import annotations

import csv
from pathlib import Path

DEFAULT_BASELINES: tuple[str, ...] = ("scripted", "max-guard", "max-attack")


def league_pool_specs(snapshots, baselines=DEFAULT_BASELINES) -> list[str]:
    """FSP pool: every past snapshot as ``ppo:<path>``, then the baselines."""
    return [f"ppo:{s}" for s in snapshots] + list(baselines)


def write_league_csv(path, rows) -> None:
    """Write the per-round tracking CSV (rewritten after every round)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_league.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add locma/envs/league.py tests/test_league.py
git commit -m "feat(league): torch-free FSP pool specs + tracking CSV writer"
```

---

### Task 2: Opponent + env builders

**Files:**
- Modify: `locma/envs/league.py`
- Test: `tests/test_league.py`

**Interfaces:**
- Consumes: `league_pool_specs`; `MixedOpponentPolicy` (`locma.policies.mixed`); `make_policy` (`locma.policies.registry`); `BattleEnv` (`locma.envs.battle_env`); `DummyVecEnv` (sb3).
- Produces:
  - `build_league_opponent(snapshots, baselines=DEFAULT_BASELINES, seed=0) -> MixedOpponentPolicy`
  - `_league_env(opponent, seed, obs_mode="token", both_seat=True) -> DummyVecEnv`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_league.py  (append)
import pytest  # noqa: E402


def test_build_league_opponent_pool_length():
    pytest.importorskip("stable_baselines3")
    from locma.envs.league import build_league_opponent

    # snapshot paths need not exist: MaskablePPOBattlePolicy loads lazily.
    opp = build_league_opponent(["a.zip", "b.zip"], ("scripted", "max-guard"), seed=0)
    assert len(opp.pool) == 4  # 2 snapshots + 2 baselines
    assert opp.name == "league"


def test_league_env_is_single_token_env():
    pytest.importorskip("stable_baselines3")
    from gymnasium import spaces

    from locma.envs.league import _league_env, build_league_opponent

    opp = build_league_opponent([], ("scripted",), seed=0)  # baselines-only pool
    env = _league_env(opp, seed=0, obs_mode="token")
    try:
        assert env.num_envs == 1
        assert isinstance(env.observation_space, spaces.Dict)
    finally:
        env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra ml --extra dev pytest tests/test_league.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_league_opponent'`

- [ ] **Step 3: Write minimal implementation**

Append to `locma/envs/league.py`:

```python
def build_league_opponent(snapshots, baselines=DEFAULT_BASELINES, seed=0):
    """Build the FSP opponent as a Python object (pool = snapshots + baselines).

    Constructed directly rather than via a spec string because snapshot paths
    can't ride a flat opponent spec (they contain colons/backslashes). Runs
    single-env, so no pickling across processes is needed.
    """
    from locma.policies.mixed import MixedOpponentPolicy  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    specs = league_pool_specs(snapshots, baselines)
    pool = [make_policy(s) for s in specs]
    return MixedOpponentPolicy(pool, name="league", seed=seed)


def _league_env(opponent, seed, obs_mode="token", both_seat=True):
    """Single-env DummyVecEnv wrapping BattleEnv with a direct opponent object."""
    import functools  # noqa: PLC0415

    from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: PLC0415

    from locma.envs.battle_env import BattleEnv  # noqa: PLC0415

    fn = functools.partial(
        BattleEnv,
        opponent=opponent,
        seed=seed,
        agent_seat=0,
        seat_random=both_seat,
        obs_mode=obs_mode,
    )
    return DummyVecEnv([fn])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra ml --extra dev pytest tests/test_league.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add locma/envs/league.py tests/test_league.py
git commit -m "feat(league): build FSP opponent object + single-env factory"
```

---

### Task 3: `avg_hard3_spec` (generic eval) in ar_study

**Files:**
- Modify: `locma/harness/ar_study.py`
- Test: `tests/test_ar_study.py`

**Interfaces:**
- Consumes: `run_match` (`locma.harness.match`); `make_policy` (`locma.policies.registry`); `HARD3` (this module).
- Produces:
  - `avg_hard3_spec(policy_spec: str, seeds, games_per_seed: int = 2) -> np.ndarray`
  - `hard3_per_seed(model_path, seeds, games_per_seed=2)` now delegates to `avg_hard3_spec(f"ppo:{model_path}", …)` (behavior unchanged).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ar_study.py  (append)
def test_avg_hard3_spec_scripted_shape_and_range():
    from locma.harness.ar_study import avg_hard3_spec

    out = avg_hard3_spec("scripted", seeds=[10, 11, 12], games_per_seed=1)
    assert out.shape == (3,)
    assert ((out >= 0.0) & (out <= 1.0)).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_ar_study.py::test_avg_hard3_spec_scripted_shape_and_range -q`
Expected: FAIL — `ImportError: cannot import name 'avg_hard3_spec'`

- [ ] **Step 3: Write minimal implementation**

In `locma/harness/ar_study.py`, add `avg_hard3_spec` and make `hard3_per_seed` delegate. Replace the current `hard3_per_seed` body:

```python
def avg_hard3_spec(policy_spec: str, seeds, games_per_seed: int = 2) -> np.ndarray:
    """avg-hard3 per seed for ANY policy spec (e.g. 'ppo:x.zip',
    'netdmcts:8,40,1.5,x.zip'). One value per seed: the mean win-rate over the
    three hard opponents, paired across models when the same seeds are used."""
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    pol = make_policy(policy_spec)
    out = np.zeros(len(seeds), dtype=np.float64)
    for i, s in enumerate(seeds):
        rates = [
            run_match(pol, make_policy(opp), games=games_per_seed, seed=int(s)).win_rate_a
            for opp in HARD3
        ]
        out[i] = float(np.mean(rates))
    return out


def hard3_per_seed(model_path: str, seeds, games_per_seed: int = 2) -> np.ndarray:
    """avg-hard3 per seed for a saved PPO model (composed via ``ppo:<path>``)."""
    return avg_hard3_spec(f"ppo:{model_path}", seeds, games_per_seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_ar_study.py -q`
Expected: PASS (all prior ar_study tests + the new one)

- [ ] **Step 5: Commit**

```bash
git add locma/harness/ar_study.py tests/test_ar_study.py
git commit -m "feat(league): generic avg_hard3_spec for any policy spec (oracle eval)"
```

---

### Task 4: `run_league` orchestration + smoke test

**Files:**
- Modify: `locma/envs/league.py`
- Test: `tests/test_league.py`

**Interfaces:**
- Consumes: `build_league_opponent`, `_league_env`, `write_league_csv` (this module); `MaskablePPO` (sb3-contrib); `hard3_per_seed` (`locma.harness.ar_study`).
- Produces:
  - `run_league(base_path, rounds=6, steps_per_round=200_000, out_dir="runs/league", seed=0, baselines=DEFAULT_BASELINES, eval_seeds=150, eval_base_seed=1_000_000, games_per_seed=2, target_kl=0.025, obs_mode="token", verbose=0) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_league.py  (append)
def test_smoke_league_two_rounds(tmp_path):
    pytest.importorskip("stable_baselines3")
    from locma.envs.league import run_league
    from locma.envs.training import train_agent

    base = str(tmp_path / "round0.zip")
    train_agent(
        "random", steps=400, out=base, seed=0, verbose=0,
        both_seat=False, obs_mode="token",
    )
    rows = run_league(
        base, rounds=2, steps_per_round=300, out_dir=str(tmp_path / "lg"),
        seed=0, eval_seeds=2, games_per_seed=1, verbose=0,
    )
    assert [r["round"] for r in rows] == [0, 1, 2]
    assert (tmp_path / "lg" / "round1.zip").exists()
    assert (tmp_path / "lg" / "round2.zip").exists()
    assert (tmp_path / "lg" / "league.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra ml --extra dev pytest tests/test_league.py::test_smoke_league_two_rounds -q`
Expected: FAIL — `ImportError: cannot import name 'run_league'`

- [ ] **Step 3: Write minimal implementation**

Append to `locma/envs/league.py`:

```python
def run_league(
    base_path,
    rounds: int = 6,
    steps_per_round: int = 200_000,
    out_dir: str = "runs/league",
    seed: int = 0,
    baselines=DEFAULT_BASELINES,
    eval_seeds: int = 150,
    eval_base_seed: int = 1_000_000,
    games_per_seed: int = 2,
    target_kl: float = 0.025,
    obs_mode: str = "token",
    verbose: int = 0,
) -> list:
    """Run an FSP league: continue the base model across `rounds`, each vs a
    growing pool of past snapshots + baselines; track avg-hard3 per round.

    Returns the list of per-round row dicts (also written to <out_dir>/league.csv).
    """
    from sb3_contrib import MaskablePPO  # noqa: PLC0415

    from locma.harness.ar_study import hard3_per_seed  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_list = [eval_base_seed + i for i in range(eval_seeds)]

    def _eval(path):
        return float(hard3_per_seed(path, eval_list, games_per_seed).mean())

    snapshots = [str(base_path)]
    rows = [
        {"round": 0, "snapshot": str(base_path), "avg_hard3": _eval(base_path), "n_seeds": eval_seeds}
    ]
    write_league_csv(out / "league.csv", rows)
    if verbose:
        print(f"[league] round 0 (base) avg_hard3={rows[0]['avg_hard3']:.4f}")

    model = MaskablePPO.load(base_path)
    model.target_kl = target_kl
    for r in range(1, rounds + 1):
        opp = build_league_opponent(snapshots, baselines, seed=seed + r)
        env = _league_env(opp, seed + r, obs_mode=obs_mode)
        model.set_env(env)
        model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
        snap = str(out / f"round{r}.zip")
        model.save(snap)
        env.close()
        snapshots.append(snap)
        rows.append(
            {"round": r, "snapshot": snap, "avg_hard3": _eval(snap), "n_seeds": eval_seeds}
        )
        write_league_csv(out / "league.csv", rows)
        if verbose:
            print(f"[league] round {r} avg_hard3={rows[-1]['avg_hard3']:.4f}")
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra ml --extra dev pytest tests/test_league.py::test_smoke_league_two_rounds -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add locma/envs/league.py tests/test_league.py
git commit -m "feat(league): run_league round loop + per-round avg-hard3 tracking"
```

---

### Task 5: `selfplay-league` + `hard3-eval` CLI

**Files:**
- Modify: `locma/cli/app.py`
- Test: `tests/test_league.py`

**Interfaces:**
- Consumes: `run_league` (`locma.envs.league`); `avg_hard3_spec` (`locma.harness.ar_study`); `typer`, `Table`, `console` (existing in `app.py`).
- Produces: two typer commands, `selfplay-league` and `hard3-eval`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_league.py  (append)
def test_selfplay_league_rejects_bad_rounds():
    from typer.testing import CliRunner

    from locma.cli.app import app

    res = CliRunner().invoke(app, ["selfplay-league", "--base", "x.zip", "--rounds", "0"])
    assert res.exit_code != 0


def test_hard3_eval_help_lists_spec():
    from typer.testing import CliRunner

    from locma.cli.app import app

    res = CliRunner().invoke(app, ["hard3-eval", "--help"])
    assert res.exit_code == 0
    assert "--spec" in res.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_league.py::test_hard3_eval_help_lists_spec -q`
Expected: FAIL — no such command `hard3-eval`

- [ ] **Step 3: Write minimal implementation**

In `locma/cli/app.py`, add both commands (place after the `ar-eval` command):

```python
@app.command("selfplay-league")
def selfplay_league_cmd(
    base: str = typer.Option(..., help="path to the token base model .zip (round 0)"),
    rounds: int = typer.Option(6, help="number of league rounds"),
    steps_per_round: int = typer.Option(200_000, help="training steps per round"),
    out_dir: str = typer.Option("runs/league", help="output dir for snapshots + league.csv"),
    seed: int = typer.Option(0),
    eval_seeds: int = typer.Option(150, help="held-out eval seeds per round"),
    eval_base_seed: int = typer.Option(1_000_000, help="first eval seed (held-out range)"),
    games_per_seed: int = typer.Option(2, help="mirrored matches per opponent per seed"),
    target_kl: float = typer.Option(0.025, help="conservative KL cap for stable self-play"),
):
    """Run FSP league self-play for the token net; track avg-hard3 per round."""
    if rounds < 1:
        raise typer.BadParameter("rounds must be >= 1")
    if steps_per_round < 1:
        raise typer.BadParameter("steps-per-round must be >= 1")
    from locma.envs.league import run_league  # noqa: PLC0415

    rows = run_league(
        base,
        rounds=rounds,
        steps_per_round=steps_per_round,
        out_dir=out_dir,
        seed=seed,
        eval_seeds=eval_seeds,
        eval_base_seed=eval_base_seed,
        games_per_seed=games_per_seed,
        target_kl=target_kl,
        verbose=1,
    )
    table = Table(title="league self-play (avg-hard3 per round)")
    table.add_column("round")
    table.add_column("avg_hard3", justify="right")
    table.add_column("snapshot")
    for row in rows:
        table.add_row(str(row["round"]), f"{row['avg_hard3']:.4f}", row["snapshot"])
    console.print(table)


@app.command("hard3-eval")
def hard3_eval_cmd(
    spec: str = typer.Option(..., help="policy spec, e.g. 'ppo:x.zip' or 'netdmcts:8,40,1.5,x.zip'"),
    seeds: int = typer.Option(100, help="number of held-out eval seeds"),
    base_seed: int = typer.Option(1_000_000, help="first eval seed (held-out range)"),
    games_per_seed: int = typer.Option(2, help="mirrored matches per opponent per seed"),
):
    """Print avg-hard3 for any policy spec (used for the netdmcts-oracle downstream)."""
    from locma.harness.ar_study import avg_hard3_spec  # noqa: PLC0415

    seed_list = [base_seed + i for i in range(seeds)]
    arr = avg_hard3_spec(spec, seed_list, games_per_seed)
    console.print(f"avg-hard3({spec}) = {float(arr.mean()):.4f} over {seeds} seeds")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_league.py -q`
Expected: PASS (all league tests)

- [ ] **Step 5: Commit**

```bash
git add locma/cli/app.py tests/test_league.py
git commit -m "feat(league): selfplay-league + hard3-eval CLI commands"
```

---

### Task 6: Full-suite green + lint gate

**Files:** none (verification).

- [ ] **Step 1: Run the CI gate**

Run:
```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra ml --extra dev pytest -q
```
Expected: ruff clean; all tests pass (new league tests + the pre-existing suite).

- [ ] **Step 2: Fix any lint/format issues, re-stage, commit if anything changed**

```bash
git add -A
git commit -m "chore(league): lint/format cleanup"
```

(If nothing changed, skip.)

---

## Execution Runbook (faster PC, after Tasks 1–6 are green)

> Detect the device first: `uv run --extra ml python -c "import torch;print(torch.cuda.is_available())"`.
> CUDA/MPS speeds the token attention modestly; CPU also works. Run the long
> steps in the background. Eval seeds live in the held-out `1_000_000+` range.

### R0 — Token base (round 0), 800k

```bash
uv run --extra ml locma train-zoo --steps-per-opponent 200000 --obs-mode token \
  --out runs/league/round0.zip --seed 0
```
This is the standard token zoo curriculum (greedy→scripted→max-guard→max-attack), 800k total.

### R1 — 6-round FSP league

```bash
uv run --extra ml locma selfplay-league --base runs/league/round0.zip \
  --rounds 6 --steps-per-round 200000 --out-dir runs/league --seed 0 --eval-seeds 150
```
Produces `runs/league/round1.zip … round6.zip` and `runs/league/league.csv`
(round, snapshot, avg_hard3, n_seeds), rewritten after each round. Watch the curve:
does it clear the prior plateau **0.639**, and does it compound or flatten?

### R2 — Precise paired verdict (best league net vs base)

Read `runs/league/league.csv`, pick the highest-`avg_hard3` round `B`, then:
```bash
uv run --extra ml locma ar-eval --flat runs/league/round0.zip \
  --ar runs/league/roundB.zip --seeds 300 --games-per-seed 2
```
Reports the paired delta + bootstrap CI of the best league net over the base.

### R3 — Oracle downstream (the real payoff)

Plug the best league net in as the netdmcts oracle and eval avg-hard3 vs the
current 0.817 (netdmcts is ~13s/game — keep seeds modest):
```bash
uv run --extra ml locma hard3-eval --spec "netdmcts:8,40,1.5,runs/league/roundB.zip" \
  --seeds 40 --games-per-seed 1
```
Compare to the frozen-oracle 0.817 (selfplay-r2). A meaningful lift is the win.

### R4 — Record the result

Append a dated entry to `docs/worklog.md`: the per-round league curve (from
`league.csv`), the R2 paired verdict vs base and vs the prior plateau 0.639, and
the R3 oracle avg-hard3 vs 0.817. Commit:
```bash
git add docs/worklog.md
git commit -m "docs(worklog): league self-play curve + netdmcts-oracle downstream"
```

---

## Self-Review

**Spec coverage:** FSP league loop → Tasks 1,2,4. Object-opponent + n_envs=1 constraint → Task 2 (`build_league_opponent`, `_league_env`). Continue-model + target_kl=0.025 → Task 4 (`run_league`). Per-round avg-hard3 tracking + CSV → Tasks 1,4. Token net / 800k base → Runbook R0. 6×200k → Runbook R1 (args, not hardcoded). Oracle downstream → Task 3 (`avg_hard3_spec`) + Task 5 (`hard3-eval`) + Runbook R3. Success criteria (vs 0.639, compound?, vs 0.817) → Runbook R1/R2/R3. Worklog → R4. Testing items 1–4 → Tasks 1,2,4,5. All covered.

**Placeholder scan:** no TBD/TODO; every code step is complete; no "similar to" references.

**Type consistency:** `league_pool_specs`/`write_league_csv`/`build_league_opponent`/`_league_env`/`run_league` signatures match between their interface blocks, implementations, tests, and the CLI. `avg_hard3_spec(spec, seeds, games_per_seed)` matches between Task 3, the `hard3_per_seed` delegation, and `hard3-eval`. Row-dict keys (`round`, `snapshot`, `avg_hard3`, `n_seeds`) are consistent between `run_league`, `write_league_csv` tests, and the CLI table. `netdmcts:8,40,1.5,<path>` spec matches the registry format.
