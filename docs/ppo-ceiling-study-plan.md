# PPO Ceiling Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling to rigorously settle whether the reactive PPO policy's ~0.60 avg-hard3 plateau is its true ceiling — in-training win-rate telemetry, an Optuna hyperparameter sweep with pruning, a paired-difference verdict harness, obs-encoding variants, and a PufferLib throughput spike — then run the study and record a symmetric verdict.

**Architecture:** All additive behind flags on top of the existing sb3-contrib MaskablePPO trainer. Phase 0 builds the tooling (HP plumbing, telemetry callback, Optuna driver, verdict harness, puffer spike). Phase 1 runs the HP sweep on the token net and decides. Phase 2 runs obs-encoding variants on the same ruler. The sb3 model artifact and the `netdmcts` oracle are never touched.

**Tech Stack:** Python ≥3.11, stable-baselines3 / sb3-contrib (MaskablePPO), PyTorch, gymnasium, Optuna (new), TensorBoard (new), numpy/scipy, typer CLI, pytest, ruff, uv.

**Companion spec:** `docs/ppo-ceiling-study-design.md` (read it first — it carries the rationale and the locked decisions).

## Global Constraints

- **sb3 only.** No PufferLib in the training/eval path; PufferLib appears solely in the Gate-0 throughput script. The sb3 model artifact format must stay loadable by the existing `netdmcts` `NetOracle` and `MaskablePPOBattlePolicy`.
- **Additive behind flags.** Every change defaults to current behavior. The flat-obs MlpPolicy path must stay byte-identical when no new knob is set. Existing tests must keep passing untouched.
- **B0 baseline (the bar):** `learning_rate=1e-4`, `target_kl=0.025`, all other PPO knobs at SB3 defaults, token obs V0, the fixed zoo curriculum (`greedy, scripted, max-guard, max-attack`), `both_seat=True`, from scratch ×3 seeds at the full **800k-step** budget (200k/opponent × 4).
- **Metric:** avg-hard3 = mean win-rate vs {`scripted`, `max-guard`, `max-attack`}, deterministic policy, held-out eval seeds (the `1_000_000+` range).
- **Decision rule (symmetric):** a candidate wins iff `mean Δ ≥ +0.03` avg-hard3 over B0 **and** the 95% paired-bootstrap CI excludes 0 → *headroom found*; otherwise → *ceiling confirmed robust to HP*.
- **Budget split:** reduced trial budget ~75–100k/opponent (pruned early); full 800k retrain only for the 3–5 survivors + B0.
- **New `sweep` extra:** `optuna>=3.6`, `tensorboard>=2.16` (SQLite is stdlib).
- **CI discipline:** `ruff check .` + `ruff format --check .` + `pytest -q` on `--extra dev`; format edits staged; run everything via `uv run` (global Python lacks the ML stack). ML-only tests guard with `pytest.importorskip`.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `pyproject.toml` | add the `sweep` optional-dependency group | modify |
| `locma/envs/training.py` | thread the full PPO knob set + `device` + token-arch kwargs + `callback`/`tensorboard_log` through `_make_model`/`train_agent`/`train_zoo` | modify |
| `locma/cli/app.py` | expose new knobs on `train`/`train-zoo`; add `sweep`, `ceiling-eval` commands | modify |
| `locma/policies/ppo.py` | let `MaskablePPOBattlePolicy` wrap a live in-memory model; infer obs variant from the model's scalar dim | modify |
| `locma/envs/eval_callback.py` | `WinRateEvalCallback` — in-training avg-hard3 → TensorBoard + Optuna prune | create |
| `locma/envs/sweep.py` | `PPOConfig` dataclass, Optuna config space, objective, study driver | create |
| `locma/harness/ceiling_eval.py` | paired-difference eval + bootstrap CI + verdict (reuses `run_match`) | create |
| `locma/envs/encode.py` | obs variant `v1` (symmetric-threat scalars); variant-parametrized encoder + space | modify |
| `locma/envs/extractor.py` | read scalar dim from the obs space (variant-agnostic) | modify |
| `scripts/puffer_bench.py` | Gate-0 throughput benchmark (sb3 vs PufferLib) | create |
| `tests/test_*` | one test module per new unit | create |
| `docs/worklog.md`, `docs/ppo-review.md` | Gate-0 SPS table + Phase-1/Phase-2 verdicts | modify (runbook) |

---

## Task 1: PPO hyperparameter plumbing

Expose every SB3 PPO knob (today only 3 of ~12 are reachable), plus `device`, token-extractor arch kwargs, and a training `callback`/`tensorboard_log`, threaded through `_make_model`/`train_agent`/`train_zoo` and the CLI. Defaults preserve current behavior exactly.

**Files:**
- Modify: `locma/envs/training.py:68-107` (`_make_model`), `110-172` (`train_agent`), `182-230` (`train_zoo`)
- Modify: `locma/cli/app.py:313-407` (`train`, `train-zoo`)
- Test: `tests/test_training_hparams.py`

**Interfaces:**
- Produces: `_make_model(env, *, obs_mode, seed, verbose, ent_coef, learning_rate=3e-4, target_kl=None, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5, device="auto", extractor_kwargs=None)` → `MaskablePPO`.
- Produces: `train_agent(...)` and `train_zoo(...)` gain the same keyword set plus `callback=None`, `tensorboard_log=None`, and (for `train_zoo`) `n_envs=1`. All forwarded to `_make_model` and to every `model.learn(..., callback=callback)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training_hparams.py
import pytest

pytest.importorskip("sb3_contrib")  # ML-only

from locma.envs.training import _build_env, _make_model


def _model(obs_mode, **hp):
    env = _build_env("random", seed=0, n_envs=1, both_seat=True, obs_mode=obs_mode)
    return _make_model(env, obs_mode=obs_mode, seed=0, verbose=0, ent_coef=0.02, **hp)


def test_make_model_threads_all_hyperparameters():
    m = _model(
        "token",
        learning_rate=1e-4,
        target_kl=0.025,
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.9,
        clip_range=0.1,
        vf_coef=1.0,
        max_grad_norm=1.0,
    )
    assert m.n_steps == 1024
    assert m.batch_size == 128
    assert m.n_epochs == 5
    assert m.gamma == 0.995
    assert m.gae_lambda == 0.9
    assert m.vf_coef == 1.0
    assert m.max_grad_norm == 1.0
    assert m.target_kl == 0.025
    # learning_rate and clip_range are stored as SB3 schedules (callables):
    assert abs(m.clip_range(1.0) - 0.1) < 1e-9
    assert abs(m.lr_schedule(1.0) - 1e-4) < 1e-9


def test_token_extractor_kwargs_applied():
    m = _model("token", extractor_kwargs={"d_model": 128, "n_layers": 1})
    ek = m.policy.features_extractor_kwargs
    assert ek["d_model"] == 128 and ek["n_layers"] == 1


def test_flat_defaults_unchanged():
    m = _model("flat")
    assert m.n_steps == 2048 and m.batch_size == 64 and m.n_epochs == 10
    assert m.gamma == 0.99 and m.gae_lambda == 0.95 and m.vf_coef == 0.5
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run --extra ml --extra dev pytest tests/test_training_hparams.py -v`
Expected: FAIL — `_make_model() got an unexpected keyword argument 'n_steps'`.

- [ ] **Step 3: Implement — rewrite `_make_model`**

```python
def _make_model(
    env,
    *,
    obs_mode: str,
    seed: int,
    verbose: int,
    ent_coef: float,
    learning_rate: float = 3e-4,
    target_kl: float | None = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "auto",
    extractor_kwargs: dict | None = None,
    tensorboard_log: str | None = None,
):
    """Construct a MaskablePPO model, selecting the policy class by obs_mode.

    All PPO knobs are explicit so a sweep can set them; defaults match SB3's own
    defaults, so an unset knob reproduces the pre-sweep behavior byte-for-byte.
    """
    from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

    common = dict(
        verbose=verbose,
        seed=seed,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        target_kl=target_kl,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        device=device,
        tensorboard_log=tensorboard_log,
    )

    if obs_mode == "token":
        from locma.envs.extractor import TokenSetExtractor  # noqa: PLC0415

        pk = dict(features_extractor_class=TokenSetExtractor)
        if extractor_kwargs:
            pk["features_extractor_kwargs"] = dict(extractor_kwargs)
        return MaskablePPO("MultiInputPolicy", env, policy_kwargs=pk, **common)

    # Default: flat obs → MlpPolicy (byte-identical to the pre-PPO2 baseline).
    return MaskablePPO("MlpPolicy", env, **common)
```

- [ ] **Step 4: Thread the knobs through `train_agent` and `train_zoo`**

Add the same keyword block (`n_steps … device`, `extractor_kwargs=None`) plus `callback=None`, `tensorboard_log=None` to both `train_agent` and `train_zoo` signatures, forward them into the `_make_model(...)` calls, and pass `callback=callback` into **every** `model.learn(...)` call. Also add `n_envs: int = 1` to `train_zoo` and replace its three hardcoded `_build_env(opp, seed, 1, ...)` calls with `_build_env(opp, seed, n_envs, ...)`.

Example for `train_zoo`'s learn calls:

```python
    for i, opp in enumerate(opps):
        if i > 0:
            model.set_env(_build_env(opp, seed, n_envs, both_seat=both_seat, obs_mode=obs_mode))
        model.learn(
            total_timesteps=steps_per_opponent,
            reset_num_timesteps=(i == 0),
            callback=callback,
        )
```

- [ ] **Step 5: Surface the new knobs on the CLI**

In `locma/cli/app.py`, add typer options to `train` and `train-zoo` for `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `vf_coef`, `max_grad_norm` (defaults matching SB3), `device` (default `"auto"`), and `tensorboard_log` (default `None`). Forward them into the `train_agent` / `train_zoo` calls. (The sweep drives training programmatically, so CLI parity is for manual B0 / control runs.)

- [ ] **Step 6: Run the tests — expect PASS**

Run: `uv run --extra ml --extra dev pytest tests/test_training_hparams.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Regression — existing training still imports/builds**

Run: `uv run --extra ml --extra dev pytest tests/ -q -k "train or env"`
Expected: PASS (no regressions in existing training/env tests).

- [ ] **Step 8: Commit**

```bash
git add locma/envs/training.py locma/cli/app.py tests/test_training_hparams.py
git commit -m "feat(ppo): expose full SB3 hyperparameter set + token arch kwargs through trainer"
```

---

## Task 2: Add the `sweep` optional-dependency group

**Files:**
- Modify: `pyproject.toml:21-24`
- Test: `tests/test_sweep_extra.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_extra.py
import tomllib
from pathlib import Path


def test_sweep_extra_declares_optuna_and_tensorboard():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    sweep = data["project"]["optional-dependencies"]["sweep"]
    joined = " ".join(sweep)
    assert "optuna" in joined
    assert "tensorboard" in joined
```

- [ ] **Step 2: Run it — expect FAIL** (`KeyError: 'sweep'`).

Run: `uv run --extra dev pytest tests/test_sweep_extra.py -v`

- [ ] **Step 3: Add the extra**

```toml
[project.optional-dependencies]
ml = ["gymnasium>=0.29", "stable-baselines3>=2.3", "sb3-contrib>=2.3", "torch>=2.2", "trueskill>=0.4.5"]
dev = ["pytest>=8", "ruff>=0.6", "httpx>=0.27"]
server = ["fastapi>=0.110", "uvicorn>=0.29"]
sweep = ["optuna>=3.6", "tensorboard>=2.16"]
```

- [ ] **Step 4: Sync + run — expect PASS**

Run: `uv sync --extra ml --extra dev --extra sweep && uv run --extra dev pytest tests/test_sweep_extra.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_sweep_extra.py
git commit -m "build: add sweep extra (optuna, tensorboard) for the PPO ceiling study"
```

---

## Task 3: `WinRateEvalCallback` + live-model battle policy

A `MaskableEvalCallback`-style callback that every `eval_freq` steps plays paired games vs {scripted, max-guard, max-attack}, logs avg-hard3 + per-opponent to TensorBoard, and (optionally) reports to an Optuna trial for pruning. To avoid disk I/O per eval, teach `MaskablePPOBattlePolicy` to wrap a live in-memory model.

**Files:**
- Modify: `locma/policies/ppo.py:22-49`
- Create: `locma/envs/eval_callback.py`
- Test: `tests/test_eval_callback.py`

**Interfaces:**
- Consumes: `locma.harness.match.run_match(pa, pb, games, seed) -> res` with `res.win_rate_a`; `locma.policies.composer.Composer`; `locma.policies.drafts.BalancedDraftPolicy`; `locma.policies.registry.make_policy`.
- Produces: `MaskablePPOBattlePolicy(model_path="model.zip", name="ppo", deterministic=True, model=None)` — if `model` is given, `_ensure` uses it instead of loading from disk.
- Produces: `WinRateEvalCallback(eval_opponents=("scripted","max-guard","max-attack"), eval_freq=50_000, n_games=120, eval_seed=1_000_000, trial=None, verbose=0)`; attribute `last_avg_hard3: float | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_callback.py
import pytest

pytest.importorskip("sb3_contrib")

from locma.envs.eval_callback import WinRateEvalCallback
from locma.envs.training import _build_env, _make_model


def test_callback_logs_avg_hard3_during_short_training():
    env = _build_env("random", seed=0, n_envs=1, both_seat=True, obs_mode="flat")
    model = _make_model(env, obs_mode="flat", seed=0, verbose=0, ent_coef=0.02)
    cb = WinRateEvalCallback(eval_freq=400, n_games=2, eval_seed=1_000_000)
    model.learn(total_timesteps=900, callback=cb)
    assert cb.last_avg_hard3 is not None
    assert 0.0 <= cb.last_avg_hard3 <= 1.0
    # TensorBoard-bound scalar names were recorded at least once:
    assert "eval/avg_hard3" in cb.logged_keys
```

- [ ] **Step 2: Run it — expect FAIL** (module missing).

Run: `uv run --extra ml --extra dev pytest tests/test_eval_callback.py -v`

- [ ] **Step 3: Teach `MaskablePPOBattlePolicy` to wrap a live model**

In `locma/policies/ppo.py`, change `__init__` to accept `model=None` and short-circuit `_ensure`:

```python
    def __init__(
        self,
        model_path: str = "model.zip",
        name: str = "ppo",
        deterministic: bool = True,
        model=None,
    ):
        self.model_path = model_path
        self.name = name
        self.deterministic = deterministic
        self._model = model  # if provided, skip the lazy file load

    def _ensure(self) -> None:
        if self._model is None:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

            self._model = MaskablePPO.load(self.model_path)
```

- [ ] **Step 4: Implement the callback**

```python
# locma/envs/eval_callback.py
"""In-training win-rate telemetry for the PPO ceiling study (requires [ml]).

Every ``eval_freq`` steps, plays paired games with the *current* policy vs a fixed
set of scripted opponents, logs avg-hard3 + per-opponent rates to TensorBoard, and
(optionally) reports the score to an Optuna trial so hopeless runs prune early.
Reuses the registry + run_match path for eval fidelity; the PPO net is piloted with
the BalancedDraftPolicy (the deployment pairing).
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class WinRateEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_opponents: tuple[str, ...] = ("scripted", "max-guard", "max-attack"),
        eval_freq: int = 50_000,
        n_games: int = 120,
        eval_seed: int = 1_000_000,
        trial=None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_opponents = eval_opponents
        self.eval_freq = eval_freq
        self.n_games = n_games
        self.eval_seed = eval_seed
        self.trial = trial
        self.last_avg_hard3: float | None = None
        self.logged_keys: set[str] = set()

    def _eval_policy(self):
        from locma.policies.composer import Composer  # noqa: PLC0415
        from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
        from locma.policies.ppo import MaskablePPOBattlePolicy  # noqa: PLC0415

        battle = MaskablePPOBattlePolicy(model=self.model, deterministic=True)
        return Composer(battle, BalancedDraftPolicy(), name="ppo")

    def _evaluate(self) -> float:
        from locma.harness.match import run_match  # noqa: PLC0415
        from locma.policies.registry import make_policy  # noqa: PLC0415

        me = self._eval_policy()
        rates = []
        for opp in self.eval_opponents:
            res = run_match(me, make_policy(opp), games=self.n_games, seed=self.eval_seed)
            wr = res.win_rate_a
            rates.append(wr)
            key = f"eval/vs_{opp.replace('-', '_')}"
            self.logger.record(key, wr)
            self.logged_keys.add(key)
        avg = sum(rates) / len(rates)
        self.logger.record("eval/avg_hard3", avg)
        self.logged_keys.add("eval/avg_hard3")
        return avg

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        avg = self._evaluate()
        self.last_avg_hard3 = avg
        self.logger.dump(self.num_timesteps)
        if self.trial is not None:
            import optuna  # noqa: PLC0415

            self.trial.report(avg, self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True

    def _on_training_end(self) -> None:
        # Guarantee at least one eval even if no step hit the modulus boundary.
        if self.last_avg_hard3 is None:
            self.last_avg_hard3 = self._evaluate()
```

Note: the `% eval_freq` check can skip if `n_envs` makes `num_timesteps` step over the exact multiple; `_on_training_end` is the backstop so `last_avg_hard3` is always set. For the sweep, set `eval_freq` to a multiple of `n_steps * n_envs`.

- [ ] **Step 5: Run the tests — expect PASS**

Run: `uv run --extra ml --extra dev pytest tests/test_eval_callback.py tests/test_ppo_policy.py -v`
Expected: PASS (callback test + any existing ppo-policy tests still green).

- [ ] **Step 6: Commit**

```bash
git add locma/envs/eval_callback.py locma/policies/ppo.py tests/test_eval_callback.py
git commit -m "feat(ppo): WinRateEvalCallback for in-training avg-hard3 telemetry + pruning"
```

---

## Task 4: `PPOConfig` + Optuna config space (pure)

**Files:**
- Create: `locma/envs/sweep.py` (config portion)
- Test: `tests/test_sweep_config.py`

**Interfaces:**
- Produces: `@dataclass PPOConfig` with fields `learning_rate, target_kl, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm, d_model, n_layers, n_heads, features_dim`; method `to_train_kwargs() -> dict` returning the keyword set Task 1's `train_zoo` accepts (arch fields packed into `extractor_kwargs`).
- Produces: `sample_config(trial, *, sweep_arch: bool = False) -> PPOConfig` — samples Phase-1a knobs; arch knobs only when `sweep_arch` (Phase-1b), else fixed at the current defaults.
- Produces: `B0_CONFIG: PPOConfig` (the baseline point) and `valid(cfg, n_envs) -> bool` (`batch_size <= n_steps * n_envs`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_config.py
import optuna

from locma.envs.sweep import B0_CONFIG, PPOConfig, sample_config, valid


def test_b0_config_matches_spec():
    assert B0_CONFIG.learning_rate == 1e-4
    assert B0_CONFIG.target_kl == 0.025
    assert B0_CONFIG.n_steps == 2048 and B0_CONFIG.batch_size == 64


def test_sample_config_respects_validity_guard():
    # A trial that would pick batch_size > n_steps*n_envs must be flagged invalid.
    bad = PPOConfig(n_steps=1024, batch_size=4096)
    assert not valid(bad, n_envs=1)
    assert valid(PPOConfig(n_steps=2048, batch_size=64), n_envs=1)


def test_sample_config_is_in_range_and_to_kwargs_roundtrips():
    study = optuna.create_study(direction="maximize")
    cfg = sample_config(study.ask())
    assert 3e-5 <= cfg.learning_rate <= 5e-4
    assert cfg.n_steps in (1024, 2048, 4096)
    kw = cfg.to_train_kwargs()
    assert kw["n_steps"] == cfg.n_steps
    assert kw["extractor_kwargs"]["d_model"] == cfg.d_model
```

- [ ] **Step 2: Run it — expect FAIL** (module missing).

Run: `uv run --extra dev --extra sweep pytest tests/test_sweep_config.py -v`

- [ ] **Step 3: Implement the config portion of `sweep.py`**

```python
# locma/envs/sweep.py
"""Optuna hyperparameter sweep for the PPO ceiling study (requires [ml] + [sweep]).

The config layer (PPOConfig, sample_config, valid) is pure and import-safe; the
objective/driver (below) need the ML stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    target_kl: float | None = 0.025
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # token-extractor arch (Phase 1b)
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    features_dim: int = 256

    def to_train_kwargs(self) -> dict:
        d = asdict(self)
        arch = {k: d.pop(k) for k in ("d_model", "n_layers", "n_heads", "features_dim")}
        d["extractor_kwargs"] = arch
        return d


# The baseline point — enqueued into the study so TPE never re-derives it.
B0_CONFIG = PPOConfig()


def valid(cfg: PPOConfig, n_envs: int) -> bool:
    """SB3 requires the rollout buffer to divide into minibatches."""
    return cfg.batch_size <= cfg.n_steps * n_envs


def sample_config(trial, *, sweep_arch: bool = False) -> PPOConfig:
    cfg = PPOConfig(
        learning_rate=trial.suggest_float("learning_rate", 3e-5, 5e-4, log=True),
        target_kl=trial.suggest_categorical("target_kl", [0.02, 0.03, 0.05, None]),
        n_steps=trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        n_epochs=trial.suggest_int("n_epochs", 3, 10),
        gamma=trial.suggest_categorical("gamma", [0.99, 0.995, 0.999]),
        gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        ent_coef=trial.suggest_float("ent_coef", 1e-3, 5e-2, log=True),
        vf_coef=trial.suggest_categorical("vf_coef", [0.5, 1.0]),
    )
    if sweep_arch:
        cfg.d_model = trial.suggest_categorical("d_model", [64, 128])
        cfg.n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
        cfg.n_heads = trial.suggest_categorical("n_heads", [4, 8])
        cfg.features_dim = trial.suggest_categorical("features_dim", [128, 256])
    return cfg
```

- [ ] **Step 4: Run the tests — expect PASS**

Run: `uv run --extra dev --extra sweep pytest tests/test_sweep_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add locma/envs/sweep.py tests/test_sweep_config.py
git commit -m "feat(sweep): PPOConfig + Optuna config space for the ceiling study"
```

---

## Task 5: Optuna objective + study driver

**Files:**
- Modify: `locma/envs/sweep.py` (append objective + driver)
- Test: `tests/test_sweep_driver.py`

**Interfaces:**
- Consumes: `sample_config`, `valid`, `B0_CONFIG`, `to_train_kwargs`; `train_zoo`; `WinRateEvalCallback`.
- Produces: `objective(trial, *, n_envs, total_steps, eval_freq, n_games, sweep_arch, tb_root, device) -> float` (returns final avg-hard3; `optuna.TrialPruned` on prune; returns `-1.0` for invalid configs).
- Produces: `run_sweep(*, storage, study_name, n_trials, n_envs=8, total_steps=300_000, eval_freq=None, n_games=120, sweep_arch=False, tb_root="runs/tb", device="auto") -> optuna.Study` — TPE sampler + HyperbandPruner, SQLite storage, enqueues `B0_CONFIG`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_driver.py
import pytest

pytest.importorskip("sb3_contrib")
pytest.importorskip("optuna")

from pathlib import Path

from locma.envs.sweep import run_sweep


def test_run_sweep_smoke_and_resumable(tmp_path):
    db = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    # Tiny budget: 1 trial, a few hundred steps, 2 eval games — just exercises the loop.
    s1 = run_sweep(
        storage=db, study_name="smoke", n_trials=1, n_envs=1,
        total_steps=800, eval_freq=400, n_games=2, tb_root=str(tmp_path / "tb"),
    )
    assert len(s1.trials) == 1
    assert s1.trials[0].value is None or 0.0 <= s1.trials[0].value <= 1.0
    # Resumability: a second call on the same storage accumulates, not overwrites.
    s2 = run_sweep(
        storage=db, study_name="smoke", n_trials=1, n_envs=1,
        total_steps=800, eval_freq=400, n_games=2, tb_root=str(tmp_path / "tb"),
    )
    assert len(s2.trials) == 2
    assert Path(tmp_path / "study.db").exists()
```

- [ ] **Step 2: Run it — expect FAIL** (`run_sweep` undefined).

Run: `uv run --extra ml --extra dev --extra sweep pytest tests/test_sweep_driver.py -v`

- [ ] **Step 3: Implement the objective + driver (append to `sweep.py`)**

```python
def objective(
    trial,
    *,
    n_envs: int,
    total_steps: int,
    eval_freq: int,
    n_games: int,
    sweep_arch: bool,
    tb_root: str,
    device: str,
) -> float:
    from locma.envs.eval_callback import WinRateEvalCallback  # noqa: PLC0415
    from locma.envs.training import train_zoo  # noqa: PLC0415

    cfg = sample_config(trial, sweep_arch=sweep_arch)
    if not valid(cfg, n_envs):
        return -1.0  # tell TPE this region is infeasible without crashing the worker

    steps_per_opp = max(1, total_steps // 4)  # 4-opponent zoo curriculum
    cb = WinRateEvalCallback(eval_freq=eval_freq, n_games=n_games, trial=trial)
    out = f"{tb_root}/trial_{trial.number}_model.zip"
    train_zoo(
        steps_per_opponent=steps_per_opp,
        out=out,
        seed=0,
        n_envs=n_envs,
        both_seat=True,
        obs_mode="token",
        device=device,
        tensorboard_log=f"{tb_root}/trial_{trial.number}",
        callback=cb,
        **cfg.to_train_kwargs(),
    )
    return float(cb.last_avg_hard3)


def run_sweep(
    *,
    storage: str,
    study_name: str,
    n_trials: int,
    n_envs: int = 8,
    total_steps: int = 300_000,
    eval_freq: int | None = None,
    n_games: int = 120,
    sweep_arch: bool = False,
    tb_root: str = "runs/tb",
    device: str = "auto",
):
    import optuna  # noqa: PLC0415
    from optuna.pruners import HyperbandPruner  # noqa: PLC0415
    from optuna.samplers import TPESampler  # noqa: PLC0415

    # eval_freq must be a multiple of the rollout size or the modulus check skips evals.
    if eval_freq is None:
        eval_freq = max(2048, (total_steps // 6) // (2048) * 2048) or 2048

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=0),
        pruner=HyperbandPruner(),
        load_if_exists=True,
    )
    # Seed TPE with the known-good baseline point (Phase 1a only — arch is fixed there).
    if not sweep_arch and not study.trials:
        from dataclasses import asdict  # noqa: PLC0415

        b0 = asdict(B0_CONFIG)
        study.enqueue_trial(
            {k: b0[k] for k in (
                "learning_rate", "target_kl", "n_steps", "batch_size", "n_epochs",
                "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
            )}
        )

    study.optimize(
        lambda t: objective(
            t,
            n_envs=n_envs,
            total_steps=total_steps,
            eval_freq=eval_freq,
            n_games=n_games,
            sweep_arch=sweep_arch,
            tb_root=tb_root,
            device=device,
        ),
        n_trials=n_trials,
    )
    return study
```

- [ ] **Step 4: Run the smoke test — expect PASS**

Run: `uv run --extra ml --extra dev --extra sweep pytest tests/test_sweep_driver.py -v`
Expected: PASS (may take a minute — it trains ~800 steps twice).

- [ ] **Step 5: Commit**

```bash
git add locma/envs/sweep.py tests/test_sweep_driver.py
git commit -m "feat(sweep): Optuna TPE objective + Hyperband driver with SQLite resume"
```

---

## Task 6: `locma sweep` CLI command

**Files:**
- Modify: `locma/cli/app.py` (add `sweep` command)
- Test: `tests/test_sweep_cli.py`

**Interfaces:**
- Consumes: `locma.envs.sweep.run_sweep`.
- Produces: CLI `locma sweep` with options `--storage`, `--study-name`, `--n-trials`, `--n-envs`, `--total-steps`, `--n-games`, `--sweep-arch/--no-sweep-arch`, `--tb-root`, `--device`. Prints the best trial's value + params.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_cli.py
import pytest

pytest.importorskip("sb3_contrib")
pytest.importorskip("optuna")

from typer.testing import CliRunner

from locma.cli.app import app


def test_sweep_cli_smoke(tmp_path):
    db = f"sqlite:///{(tmp_path / 's.db').as_posix()}"
    r = CliRunner().invoke(
        app,
        ["sweep", "--storage", db, "--study-name", "cli", "--n-trials", "1",
         "--n-envs", "1", "--total-steps", "800", "--n-games", "2",
         "--tb-root", str(tmp_path / "tb")],
    )
    assert r.exit_code == 0, r.output
    assert "best" in r.output.lower()
```

- [ ] **Step 2: Run it — expect FAIL** (no `sweep` command).

Run: `uv run --extra ml --extra dev --extra sweep pytest tests/test_sweep_cli.py -v`

- [ ] **Step 3: Implement the command**

```python
@app.command()
def sweep(
    storage: str = typer.Option("sqlite:///runs/ceiling.db", help="Optuna storage URL"),
    study_name: str = typer.Option("ceiling-phase1a", help="Optuna study name"),
    n_trials: int = typer.Option(50, help="trials to run this invocation"),
    n_envs: int = typer.Option(8, help="parallel envs per trial"),
    total_steps: int = typer.Option(300_000, help="reduced per-trial budget (4-opp curriculum)"),
    n_games: int = typer.Option(120, help="eval games per opponent in the callback"),
    sweep_arch: bool = typer.Option(False, help="Phase 1b: sweep token-extractor arch"),
    tb_root: str = typer.Option("runs/tb", help="TensorBoard + model output root"),
    device: str = typer.Option("auto", help="torch device: auto|cpu|cuda|mps"),
):
    """Run the PPO ceiling-study hyperparameter sweep (requires [ml] + [sweep])."""
    try:
        from locma.envs.sweep import run_sweep  # noqa: PLC0415
    except ImportError as e:
        raise typer.BadParameter("sweep requires extras: uv sync --extra ml --extra sweep") from e
    study = run_sweep(
        storage=storage, study_name=study_name, n_trials=n_trials, n_envs=n_envs,
        total_steps=total_steps, n_games=n_games, sweep_arch=sweep_arch,
        tb_root=tb_root, device=device,
    )
    bt = study.best_trial
    console.print(f"best avg_hard3 = {bt.value:.3f}  (trial {bt.number})")
    console.print(f"best params: {bt.params}")
```

- [ ] **Step 4: Run — expect PASS**

Run: `uv run --extra ml --extra dev --extra sweep pytest tests/test_sweep_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add locma/cli/app.py tests/test_sweep_cli.py
git commit -m "feat(cli): locma sweep — Optuna ceiling-study sweep command"
```

---

## Task 7: `ceiling_eval` statistics (pure)

The verdict math: paired bootstrap CI over per-eval-seed avg-hard3 differences + the symmetric +0.03 decision. Pure functions, no game-running — unit-tested on synthetic inputs.

**Files:**
- Create: `locma/harness/ceiling_eval.py` (stats portion)
- Test: `tests/test_ceiling_eval_stats.py`

**Interfaces:**
- Produces: `paired_bootstrap_ci(deltas, n_boot=10_000, seed=0, alpha=0.05) -> (mean, lo, hi)`.
- Produces: `decide(mean_delta, ci_lo, ci_hi, threshold=0.03) -> str` returning `"headroom"` or `"ceiling-confirmed"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ceiling_eval_stats.py
import numpy as np

from locma.harness.ceiling_eval import decide, paired_bootstrap_ci


def test_bootstrap_ci_brackets_clear_positive_signal():
    deltas = np.full(40, 0.05)
    mean, lo, hi = paired_bootstrap_ci(deltas, n_boot=2000, seed=0)
    assert abs(mean - 0.05) < 1e-9 and lo > 0.0


def test_decide_headroom_when_above_threshold_and_ci_excludes_zero():
    assert decide(0.05, 0.02, 0.08) == "headroom"


def test_decide_ceiling_when_within_noise():
    assert decide(0.005, -0.02, 0.03) == "ceiling-confirmed"


def test_decide_ceiling_when_point_high_but_ci_crosses_zero():
    # Big point estimate but the CI includes 0 → not resolved → ceiling-confirmed.
    assert decide(0.04, -0.01, 0.09) == "ceiling-confirmed"
```

- [ ] **Step 2: Run it — expect FAIL** (module missing).

Run: `uv run --extra dev pytest tests/test_ceiling_eval_stats.py -v`

- [ ] **Step 3: Implement the stats**

```python
# locma/harness/ceiling_eval.py
"""Verdict harness for the PPO ceiling study.

Stats layer (this top half) is pure numpy — paired bootstrap CI over per-eval-seed
avg-hard3 differences + the symmetric +0.03 decision rule. The runner (bottom half)
reuses run_match to produce the deltas.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(deltas, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """Mean and (1-alpha) percentile CI of the paired differences via bootstrap.

    ``deltas[i]`` = (candidate avg-hard3 − B0 avg-hard3) at eval seed i, using common
    random numbers so the difference has much lower variance than either rate alone.
    """
    d = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot_means = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(d.mean()), float(lo), float(hi)


def decide(mean_delta: float, ci_lo: float, ci_hi: float, threshold: float = 0.03) -> str:
    """Symmetric verdict: a lift counts only if it clears the threshold AND the CI
    excludes zero. Everything else confirms the ceiling."""
    if mean_delta >= threshold and ci_lo > 0.0:
        return "headroom"
    return "ceiling-confirmed"
```

- [ ] **Step 4: Run — expect PASS**

Run: `uv run --extra dev pytest tests/test_ceiling_eval_stats.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add locma/harness/ceiling_eval.py tests/test_ceiling_eval_stats.py
git commit -m "feat(eval): ceiling-study paired-bootstrap CI + symmetric verdict"
```

---

## Task 8: `ceiling_eval` runner + `locma ceiling-eval` CLI

Produce per-eval-seed avg-hard3 for a candidate model and B0 over identical seeds, difference them, and apply the verdict.

**Files:**
- Modify: `locma/harness/ceiling_eval.py` (append runner)
- Modify: `locma/cli/app.py` (add `ceiling-eval` command)
- Test: `tests/test_ceiling_eval_runner.py`

**Interfaces:**
- Consumes: `run_match`, `Composer`, `BalancedDraftPolicy`, `MaskablePPOBattlePolicy`, `make_policy`.
- Produces: `avg_hard3_per_seed(model_path, seeds, games_per_seed, opponents=("scripted","max-guard","max-attack")) -> list[float]`.
- Produces: `run_verdict(candidate_paths, b0_paths, seeds, games_per_seed, threshold=0.03) -> dict` with keys `mean_delta, ci_lo, ci_hi, verdict, cand_avg, b0_avg` (averaging over the model lists, paired per seed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ceiling_eval_runner.py
import pytest

pytest.importorskip("sb3_contrib")

from locma.envs.training import _build_env, _make_model
from locma.harness.ceiling_eval import avg_hard3_per_seed, run_verdict


def _tiny_model(tmp_path, name):
    env = _build_env("random", seed=0, n_envs=1, both_seat=True, obs_mode="flat")
    m = _make_model(env, obs_mode="flat", seed=0, verbose=0, ent_coef=0.02)
    m.learn(total_timesteps=400)
    p = str(tmp_path / f"{name}.zip")
    m.save(p)
    return p


def test_runner_smoke_returns_verdict(tmp_path):
    cand = _tiny_model(tmp_path, "cand")
    b0 = _tiny_model(tmp_path, "b0")
    seeds = [1_000_000, 1_000_001]
    rates = avg_hard3_per_seed(cand, seeds, games_per_seed=2)
    assert len(rates) == len(seeds) and all(0.0 <= r <= 1.0 for r in rates)
    out = run_verdict([cand], [b0], seeds=seeds, games_per_seed=2)
    assert out["verdict"] in ("headroom", "ceiling-confirmed")
    assert set(out) >= {"mean_delta", "ci_lo", "ci_hi", "cand_avg", "b0_avg"}
```

- [ ] **Step 2: Run it — expect FAIL** (`avg_hard3_per_seed` undefined).

Run: `uv run --extra ml --extra dev pytest tests/test_ceiling_eval_runner.py -v`

- [ ] **Step 3: Implement the runner (append to `ceiling_eval.py`)**

```python
def _ppo_policy(model_path: str):
    from locma.policies.composer import Composer  # noqa: PLC0415
    from locma.policies.drafts import BalancedDraftPolicy  # noqa: PLC0415
    from locma.policies.ppo import MaskablePPOBattlePolicy  # noqa: PLC0415

    return Composer(MaskablePPOBattlePolicy(model_path), BalancedDraftPolicy(), name="ppo")


def avg_hard3_per_seed(
    model_path,
    seeds,
    games_per_seed,
    opponents=("scripted", "max-guard", "max-attack"),
):
    """avg-hard3 (mean win-rate over the 3 opponents) at each eval seed."""
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    me = _ppo_policy(model_path)
    opps = [make_policy(o) for o in opponents]
    out = []
    for s in seeds:
        rates = [run_match(me, opp, games=games_per_seed, seed=s).win_rate_a for opp in opps]
        out.append(sum(rates) / len(rates))
    return out


def run_verdict(candidate_paths, b0_paths, seeds, games_per_seed, threshold: float = 0.03):
    """Paired per-seed avg-hard3 difference (candidate models − B0 models), averaged
    over each arm's model list, with a bootstrap CI and the symmetric verdict."""
    import numpy as np  # noqa: PLC0415

    def arm_matrix(paths):
        # rows = models, cols = seeds
        return np.array([avg_hard3_per_seed(p, seeds, games_per_seed) for p in paths])

    cand = arm_matrix(candidate_paths).mean(axis=0)  # per-seed mean over candidate models
    b0 = arm_matrix(b0_paths).mean(axis=0)
    deltas = cand - b0
    mean_delta, lo, hi = paired_bootstrap_ci(deltas)
    return {
        "mean_delta": mean_delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "verdict": decide(mean_delta, lo, hi, threshold),
        "cand_avg": float(cand.mean()),
        "b0_avg": float(b0.mean()),
    }
```

- [ ] **Step 4: Add the `ceiling-eval` CLI command in `app.py`**

```python
@app.command("ceiling-eval")
def ceiling_eval_cmd(
    candidates: str = typer.Option(..., help="comma-separated candidate model .zip paths"),
    baselines: str = typer.Option(..., help="comma-separated B0 model .zip paths"),
    seeds: int = typer.Option(40, help="number of held-out eval seeds (from 1_000_000)"),
    games_per_seed: int = typer.Option(25, help="paired games per opponent per seed"),
    threshold: float = typer.Option(0.03, help="avg-hard3 lift required for 'headroom'"),
):
    """Rigorous paired-difference verdict for the PPO ceiling study (requires [ml])."""
    try:
        from locma.harness.ceiling_eval import run_verdict  # noqa: PLC0415
    except ImportError as e:
        raise typer.BadParameter("ceiling-eval requires the [ml] extra") from e
    seed_list = list(range(1_000_000, 1_000_000 + seeds))
    out = run_verdict(
        candidates.split(","), baselines.split(","),
        seeds=seed_list, games_per_seed=games_per_seed, threshold=threshold,
    )
    console.print(
        f"cand={out['cand_avg']:.3f}  B0={out['b0_avg']:.3f}  "
        f"Δ={out['mean_delta']:+.3f}  95% CI [{out['ci_lo']:+.3f}, {out['ci_hi']:+.3f}]"
    )
    console.print(f"[bold]VERDICT: {out['verdict']}[/]")
```

- [ ] **Step 5: Run — expect PASS**

Run: `uv run --extra ml --extra dev pytest tests/test_ceiling_eval_runner.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add locma/harness/ceiling_eval.py locma/cli/app.py tests/test_ceiling_eval_runner.py
git commit -m "feat(eval): ceiling-eval runner + CLI (paired avg-hard3 verdict)"
```

---

## Task 9: Gate-0 PufferLib throughput spike

A throughput-only benchmark (no learning) comparing sb3 `SubprocVecEnv` SPS vs PufferLib vectorization on `BattleEnv`. PufferLib is optional — the script reports sb3 numbers regardless and skips the Puffer arm if the package is absent.

**Files:**
- Create: `scripts/puffer_bench.py`
- Test: `tests/test_puffer_bench.py`

**Interfaces:**
- Produces: `sb3_sps(n_envs, steps, obs_mode="token", opponent="scripted") -> float` (steps/sec).
- Produces: `puffer_sps(n_envs, steps, ...) -> float | None` (None if PufferLib not installed).
- Produces: `main()` printing a small table.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_puffer_bench.py
import pytest

pytest.importorskip("sb3_contrib")

from scripts.puffer_bench import sb3_sps


def test_sb3_sps_positive():
    sps = sb3_sps(n_envs=1, steps=200, obs_mode="flat", opponent="random")
    assert sps > 0.0
```

- [ ] **Step 2: Run it — expect FAIL** (module missing). Ensure `scripts/__init__.py` exists or add `conftest.py` rootdir handling so `scripts` is importable; simplest is to create empty `scripts/__init__.py`.

Run: `uv run --extra ml --extra dev pytest tests/test_puffer_bench.py -v`

- [ ] **Step 3: Implement the benchmark**

```python
# scripts/puffer_bench.py
"""Gate-0 throughput spike: sb3 SubprocVecEnv vs PufferLib vectorization on BattleEnv.

Throughput only (no learning). Informs a FUTURE PufferLib migration decision; the
ceiling study runs on sb3 regardless. Uses time.perf_counter (allowed) — never call
this from a workflow script (Date/random restrictions there do not apply to scripts).
"""

from __future__ import annotations

import time

import numpy as np


def sb3_sps(n_envs: int, steps: int, obs_mode: str = "token", opponent: str = "scripted") -> float:
    from locma.envs.training import _build_env  # noqa: PLC0415

    vec = _build_env(opponent, seed=0, n_envs=n_envs, both_seat=True, obs_mode=obs_mode)
    vec.reset()
    t0 = time.perf_counter()
    n = 0
    while n < steps:
        actions = np.array([vec.action_space.sample() for _ in range(n_envs)])
        vec.step(actions)
        n += n_envs
    dt = time.perf_counter() - t0
    vec.close()
    return n / dt if dt > 0 else 0.0


def puffer_sps(n_envs: int, steps: int, obs_mode: str = "token", opponent: str = "scripted"):
    try:
        import pufferlib  # noqa: F401, PLC0415
    except ImportError:
        return None
    # PufferLib wiring is intentionally deferred to the run session on the GPU box,
    # where pufferlib is installed; this stub returns None until then so the sb3
    # number is always available. See the runbook (Gate 0).
    return None


def main() -> None:
    print(f"{'config':<22}{'SPS':>12}")
    for ne in (1, 4, 8, 16):
        print(f"sb3 token n_envs={ne:<3}{sb3_sps(ne, 4000):>12.0f}")
    p = puffer_sps(8, 4000)
    print(f"puffer token n_envs=8 {('(absent)' if p is None else f'{p:.0f}'):>12}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run — expect PASS**

Run: `uv run --extra ml --extra dev pytest tests/test_puffer_bench.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/puffer_bench.py scripts/__init__.py tests/test_puffer_bench.py
git commit -m "feat(bench): Gate-0 sb3-vs-PufferLib throughput spike (sb3 arm)"
```

---

## Task 10: Observation variant V1 (symmetric-threat scalars)

Add a `v1` token-obs variant: five extra tactical scalars giving the net symmetric threat awareness (V0 only encodes *my* guard/attack/reachable, never the opponent's threat). Make the extractor read the scalar dim from the obs space so any variant is drop-in. Variant is selected by the `obs_mode` string `"token-v1"` and inferred at eval time from the model's scalar dim.

**Files:**
- Modify: `locma/envs/encode.py` (variant-parametrize the token encoder + space; new constant `N_TACTICAL_V1=18`)
- Modify: `locma/envs/extractor.py:48-99` (read scalar dim from obs space)
- Modify: `locma/envs/battle_env.py` (accept `obs_mode="token-v1"`), `locma/envs/training.py` (`_make_battle_env`/`_build_env` pass-through already string-based — just allow the value), `locma/policies/ppo.py` (`_encode_for` infers variant from scalar dim)
- Test: `tests/test_obs_v1.py`

**Interfaces:**
- Produces: `encode_battle_tokens(view, variant="v0")` and `token_obs_space(variant="v0")`; `N_TACTICAL_V1 = 18`.
- Consumes (extractor): `observation_space["scalars"].shape[0]` instead of the `N_TACTICAL` constant.

- [ ] **Step 1: Read the current encoder** (`locma/envs/encode.py:195-327`) and `battle_env.py`'s obs-space construction, so the splice matches the real structure. (V0 scalar order is documented at `encode.py:206-219`.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_obs_v1.py
from types import SimpleNamespace

from locma.envs.encode import N_TACTICAL_V1, encode_battle_tokens, token_obs_space


def _card(card_id, atk, dfn, abilities="------", can_attack=False, has_attacked=False, cost=1):
    return SimpleNamespace(
        card_id=card_id, attack=atk, defense=dfn, abilities=abilities,
        can_attack=can_attack, has_attacked=has_attacked, cost=cost, type=0,
    )


def _view():
    # me: one 3/2 ready attacker; op: one 4/4 ready attacker with no guard.
    return SimpleNamespace(
        turn=5, me_health=10, op_health=20, me_mana=3, op_hand_count=4,
        my_hand=[_card(1, 0, 0, cost=2)],
        my_board=[_card(2, 3, 2, can_attack=True)],
        op_board=[_card(3, 4, 4, can_attack=True)],
    )


def test_v0_unchanged_length():
    s = encode_battle_tokens(_view(), variant="v0")["scalars"]
    assert s.shape[0] == 13


def test_v1_appends_five_symmetric_threat_scalars():
    s = encode_battle_tokens(_view(), variant="v1")["scalars"]
    assert s.shape[0] == N_TACTICAL_V1 == 18
    # [13]=my_guard_count=0, [14]=op_total_attack=4, [15]=op_reachable=4,
    # [16]=exposed_to_lethal (4>=10? no →0), [17]=card_advantage=(1+1)-(4+1)=-3
    assert s[13] == 0.0
    assert s[14] == 4.0
    assert s[15] == 4.0
    assert s[16] == 0.0
    assert s[17] == -3.0


def test_token_obs_space_v1_scalar_shape():
    sp = token_obs_space(variant="v1")
    assert sp["scalars"].shape == (18,)
```

- [ ] **Step 3: Run it — expect FAIL** (`N_TACTICAL_V1` / `variant` kwarg missing).

Run: `uv run --extra dev pytest tests/test_obs_v1.py -v`

- [ ] **Step 4: Implement V1 in `encode.py`**

Add near the other constants: `N_TACTICAL_V1 = 18`. Then parametrize the encoder — after building V0's `scalars` array, when `variant == "v1"` append the five extra scalars:

```python
def encode_battle_tokens(view, variant: str = "v0") -> dict:
    # ... unchanged through the V0 `scalars = np.array([...], dtype=np.float32)` build ...

    if variant == "v1":
        my_guard_count = sum(1 for c in view.my_board if c.abilities[_GUARD_IDX] != "-")
        op_total_attack = sum(float(c.attack) for c in view.op_board)
        if my_guard_count > 0:
            op_reachable = 0.0
        else:
            op_reachable = sum(
                float(c.attack) for c in view.op_board if c.can_attack and not c.has_attacked
            )
        exposed_to_lethal = 1.0 if op_reachable >= view.me_health else 0.0
        card_advantage = float(
            (len(view.my_hand) + len(view.my_board)) - (view.op_hand_count + len(view.op_board))
        )
        scalars = np.concatenate(
            [
                scalars,
                np.array(
                    [my_guard_count, op_total_attack, op_reachable, exposed_to_lethal, card_advantage],
                    dtype=np.float32,
                ),
            ]
        )

    return {"tokens": tokens, "card_ids": card_ids, "token_mask": token_mask, "scalars": scalars}
```

And parametrize the space:

```python
def token_obs_space(variant: str = "v0"):
    from gymnasium import spaces  # noqa: PLC0415

    n_scalar = N_TACTICAL_V1 if variant == "v1" else N_TACTICAL
    # ... same Dict, but the "scalars" Box uses shape=(n_scalar,) ...
```

- [ ] **Step 5: Make the extractor variant-agnostic** (`extractor.py`)

Replace the `N_TACTICAL`-based scalar MLP with a dim read from the obs space:

```python
        n_scalar = int(observation_space["scalars"].shape[0])
        self.scalar_mlp = nn.Sequential(
            nn.LayerNorm(n_scalar),
            nn.Linear(n_scalar, d_model),
            nn.ReLU(),
        )
```

(Drop the now-unused `N_TACTICAL` import if it is unused elsewhere in the file.)

- [ ] **Step 6: Wire `"token-v1"` through env + policy**

- In `battle_env.py`, accept `obs_mode in ("flat","token","token-v1")`; for `"token-v1"` build `token_obs_space("v1")` and call `encode_battle_tokens(view, "v1")`.
- In `training.py`, relax the `obs_mode` validation paths to allow `"token-v1"` (it is already passed as a string into `_make_battle_env`; `_make_model` treats any `obs_mode != "flat"` as the token MultiInputPolicy path — confirm `obs_mode.startswith("token")` selects the token branch).
- In `locma/policies/ppo.py`, infer the variant from the loaded model's scalar dim:

```python
def _encode_for(model, view):
    from gymnasium import spaces  # noqa: PLC0415

    if isinstance(model.observation_space, spaces.Dict):
        n_scalar = int(model.observation_space["scalars"].shape[0])
        variant = "v1" if n_scalar == 18 else "v0"
        return encode_battle_tokens(view, variant)
    return encode_battle(view)
```

- In `app.py`, update the `obs_mode` validation in `train`/`train-zoo` to accept `"token-v1"`.

- [ ] **Step 7: Run the tests — expect PASS**

Run: `uv run --extra ml --extra dev pytest tests/test_obs_v1.py tests/test_training_hparams.py -v`
Expected: PASS (V1 scalars correct; token training path still builds, now also for `token-v1`).

- [ ] **Step 8: Commit**

```bash
git add locma/envs/encode.py locma/envs/extractor.py locma/envs/battle_env.py locma/envs/training.py locma/policies/ppo.py locma/cli/app.py tests/test_obs_v1.py
git commit -m "feat(obs): token-v1 variant (symmetric-threat scalars); extractor reads scalar dim from obs space"
```

---

## Task 11 (OPTIONAL / STRETCH): Observation variant V2 (relational trade matrix)

Only build if Phase-2 V1 looks promising or schedule allows. Adds a per-pair (my-attacker × op-blocker) relation block — `can_A_kill_B` (`a.atk ≥ b.def`), `can_B_kill_A`, `favorable_trade` — as a flattened `MAX_BOARD × MAX_BOARD × 3` relation input fed through a small MLP and concatenated in the extractor head. Selected as `obs_mode="token-v2"`, inferred at eval time by the presence of a `"relations"` key in the obs space.

This is the `ppo-review.md` §8.4A "relational objects" lever. Decompose at build time into: (a) `encode_battle_tokens(view, "v2")` emitting the `relations` array + a `relation_obs_space("v2")`; (b) extractor `relation_mlp` + head fuse; (c) env/policy wiring; (d) a pure test asserting the trade-matrix values on a crafted 2-attacker × 2-blocker view. Mirror Task 10's structure and TDD rhythm. **Do not start without confirming with the run-session owner that V1 justified it.**

---

## Execution Runbook (operational — runs, not code)

Run on the RTX 4080 box after Tasks 1–10 are merged. Record every result in `docs/worklog.md`. These are not TDD tasks; each ends with a worklog entry, not a test.

### R0 — Gate 0: PufferLib spike

```bash
uv run --extra ml python scripts/puffer_bench.py
```

- Record the SPS table. Pick the `n_envs` that maxes sb3 SPS for the sweep; record CPU-vs-CUDA training SPS for the token net. Write a one-paragraph go/no-go for a *future* Puffer migration. **The study proceeds on sb3 regardless.**

### R1 — Train B0 (the baseline, ×3 seeds, full 800k)

```bash
for s in 0 1 2; do
  uv run --extra ml locma train-zoo --obs-mode token --learning-rate 1e-4 \
    --target-kl 0.025 --steps-per-opponent 200000 --seed $s \
    --device cuda --tensorboard-log runs/tb/b0_s$s --out runs/b0_s$s.zip
done
```

- These three `.zip` are the verdict baseline. Eyeball the TensorBoard `eval/avg_hard3` curves; B0 should settle ~0.60.

### R2 — Phase 1a: core-PPO sweep

```bash
uv run --extra ml --extra sweep locma sweep \
  --storage sqlite:///runs/ceiling.db --study-name phase1a \
  --n-trials 150 --n-envs 8 --total-steps 320000 --n-games 120 \
  --tb-root runs/tb --device cuda
```

- Run in overnight batches (re-invoke to add trials — SQLite resumes). Watch the Optuna dashboard / `eval/avg_hard3`. When trials plateau, list the top 5 distinct configs.

### R3 — Phase 1b: arch sweep around the 1a winner

Pin the 1a-best core knobs in `B0_CONFIG`-style defaults (or extend `sample_config` to fix them), then:

```bash
uv run --extra ml --extra sweep locma sweep \
  --storage sqlite:///runs/ceiling.db --study-name phase1b \
  --n-trials 40 --sweep-arch --n-envs 8 --total-steps 320000 --device cuda --tb-root runs/tb
```

### R4 — Rigorous confirm + verdict #1

Retrain the top-K (3–5) survivors at the **full 800k** ×3 seeds (as in R1, with the survivor's HPs), then:

```bash
uv run --extra ml locma ceiling-eval \
  --candidates runs/cand1_s0.zip,runs/cand1_s1.zip,runs/cand1_s2.zip \
  --baselines runs/b0_s0.zip,runs/b0_s1.zip,runs/b0_s2.zip \
  --seeds 40 --games-per-seed 25 --threshold 0.03
```

- First run `uv run --extra ml locma noise-floor ppo:runs/b0_s0.zip --games 800` and confirm resolution < 0.03. Then apply the printed VERDICT. Write **verdict #1** to `docs/worklog.md` and slot it into `docs/ppo-review.md` §8 (headroom-with-recipe → also update the shipped training defaults; ceiling-confirmed → record the null + the fANOVA importance from `optuna.importance.get_param_importances(study)`).

### R5 — Phase 2: obs V1 (and V2 if built) — verdict #2

Repeat R1+R4 with `--obs-mode token-v1` and Phase-1's best HPs (B0' = best-HP + token-v1), comparing against the same B0. Same +0.03 ruler. Write **verdict #2**.

### R6 — Flat control (small)

A short sweep (`--obs-mode flat`, ~30 trials) just to confirm the flat ceiling does/doesn't move under the same treatment; one worklog line.

---

## Self-Review (run by the plan author before handoff)

- **Spec coverage:** Gate 0 → Task 9 + R0. Telemetry (b) → Task 3. HP search (c)+(e) → Tasks 4–6 + R2/R3. Paired +0.03 verdict → Tasks 7–8 + R4. Obs (Phase 2) → Tasks 10–11 + R5. B0 baseline → R1. Flat control → R6. Offline (d) → scoped out in the spec; no task (correct). All spec sections map to a task or a runbook step.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; Task 11 is explicitly optional with a concrete decomposition.
- **Type consistency:** `to_train_kwargs()` packs arch into `extractor_kwargs`, which Task 1's `_make_model` consumes; `WinRateEvalCallback.last_avg_hard3` feeds `objective`'s return; `avg_hard3_per_seed`→`run_verdict`→`decide` chain is consistent; eval opponents tuple identical across callback and runner.
- **Known follow-ups for the implementer:** confirm the registry/`Composer` pairing matches how `ppo:` is deployed (BalancedDraftPolicy); confirm `_make_model`'s token branch is selected by `obs_mode.startswith("token")` after Task 10 (so `token-v1` routes correctly).
