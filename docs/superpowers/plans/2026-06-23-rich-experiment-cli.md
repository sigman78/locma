# Rich Experiment CLI + Repo Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the LOCM 1.2 explore kit CLI into a rich experiment surface (play+render+log, tournament+matrix+openskill, noise-floor, sprt, byte-identical replay) on a new engine trace hook, plus repo hygiene (ruff, editorconfig, docs split).

**Architecture:** A single optional `on_step` callback in `run_game` drives both a trace Recorder and a live Renderer with zero overhead when unused. Recorded games serialize to a canonical JSONL game-log with a content hash; `replay` re-runs from the seed and asserts the recomputed hash matches. openskill ratings are added alongside existing Elo. CLI orchestration stays in `cli/app.py`; pure helpers live in `harness/trace.py`, `stats/openskill_ratings.py`, `cli/render.py`.

**Tech Stack:** Python ≥3.11, Typer, rich, scipy, openskill (new core dep), ruff (new dev tool), pytest.

## Global Constraints

- Python `requires-python = ">=3.11"`; all code uses `from __future__ import annotations`.
- Core dependencies stay light: only `openskill` is added to `[project.dependencies]`. No Node toolchain.
- ML libs (gymnasium, sb3, torch, trueskill) remain behind the `[ml]` extra and must never be imported by core/CLI/harness/stats.
- Determinism is sacred: same seed + same policies → identical outcome and identical trace. Never call `random` without a seeded `Random`.
- ruff: `line-length = 100`, lint `select = ["E","F","I","UP","B","SIM","PLC"]`, `ruff format` is the formatter.
- All commands run via `uv run`. Tests live in `tests/`, run with `uv run pytest`.
- Lazy-import heavy/optional modules inside command bodies (existing pattern: `fetch_cards` import is local with `# noqa: PLC0415`).

---

### Task 1: Repo hygiene foundation — editorconfig + ruff config + fix violations

**Files:**
- Create: `.editorconfig`
- Modify: `pyproject.toml` (add `[tool.ruff]`, add `ruff` to `dev` extra)
- Modify: any source files ruff flags (run-driven)

**Interfaces:**
- Consumes: nothing.
- Produces: a lint-clean, formatted tree. No code symbols.

- [ ] **Step 1: Create `.editorconfig`**

```ini
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space

[*.py]
indent_size = 4

[*.{md,yaml,yml,json,toml}]
indent_size = 2
```

- [ ] **Step 2: Add ruff config and dev dependency to `pyproject.toml`**

Change the `dev` extra line and append ruff config at the end of the file:

```toml
dev = ["pytest>=8", "ruff>=0.6"]
```

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "PLC"]
```

- [ ] **Step 3: Sync the new dev dependency**

Run: `uv sync --extra dev`
Expected: resolves and installs `ruff`.

- [ ] **Step 4: Format the codebase ("prettify")**

Run: `uv run ruff format .`
Expected: reports files reformatted. Review the diff is whitespace/quote-only (no logic change).

- [ ] **Step 5: Auto-fix safe lint violations**

Run: `uv run ruff check . --fix`
Expected: import-sorting (`I`) and other safe fixes applied.

- [ ] **Step 6: Resolve any remaining lint warnings by hand**

Run: `uv run ruff check .`
Expected: any leftover `B`/`SIM`/`PLC`/`UP` warnings printed with file:line. Fix each at its site (e.g. add `# noqa: PLC0415` to the existing lazy imports in `cli/app.py` that are intentional, mirror the existing comment style). Re-run until output is `All checks passed!`.

- [ ] **Step 7: Confirm tests still pass after formatting**

Run: `uv run pytest -q`
Expected: `51 passed` (no behavior changed).

- [ ] **Step 8: Commit**

```bash
git add .editorconfig pyproject.toml locma tests train.py
git commit -m "chore: add editorconfig + ruff format/lint, fix violations"
```

---

### Task 2: Canonical action serialization

**Files:**
- Modify: `locma/core/actions.py`
- Test: `tests/test_action_serde.py`

**Interfaces:**
- Consumes: `Summon`, `Attack`, `Use`, `Pass`, `Action` (existing in `core/actions.py`).
- Produces:
  - `action_to_dict(action: Action) -> dict` — keys: `Summon`→`{"t":"summon","id":int}`; `Attack`→`{"t":"attack","a":int,"target":int}`; `Use`→`{"t":"use","item":int,"target":int}`; `Pass`→`{"t":"pass"}`.
  - `action_from_dict(d: dict) -> Action` — inverse.

- [ ] **Step 1: Write the failing test**

```python
from locma.core.actions import Summon, Attack, Use, Pass, action_to_dict, action_from_dict


def test_action_roundtrip():
    for action in (Summon(5), Attack(3, -1), Attack(3, 7), Use(2, -1), Use(2, 9), Pass()):
        assert action_from_dict(action_to_dict(action)) == action


def test_action_to_dict_shapes():
    assert action_to_dict(Summon(5)) == {"t": "summon", "id": 5}
    assert action_to_dict(Attack(3, -1)) == {"t": "attack", "a": 3, "target": -1}
    assert action_to_dict(Use(2, 9)) == {"t": "use", "item": 2, "target": 9}
    assert action_to_dict(Pass()) == {"t": "pass"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_action_serde.py -v`
Expected: FAIL with `ImportError: cannot import name 'action_to_dict'`.

- [ ] **Step 3: Implement in `locma/core/actions.py`**

Append below the existing `Action` alias:

```python
def action_to_dict(action: Action) -> dict:
    if isinstance(action, Summon):
        return {"t": "summon", "id": action.card_instance_id}
    if isinstance(action, Attack):
        return {"t": "attack", "a": action.attacker_id, "target": action.target_id}
    if isinstance(action, Use):
        return {"t": "use", "item": action.item_instance_id, "target": action.target_id}
    if isinstance(action, Pass):
        return {"t": "pass"}
    raise TypeError(f"unknown action: {action!r}")


def action_from_dict(d: dict) -> Action:
    t = d["t"]
    if t == "summon":
        return Summon(d["id"])
    if t == "attack":
        return Attack(d["a"], d["target"])
    if t == "use":
        return Use(d["item"], d["target"])
    if t == "pass":
        return Pass()
    raise ValueError(f"unknown action tag: {t!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_action_serde.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add locma/core/actions.py tests/test_action_serde.py
git commit -m "feat(core): canonical action_to_dict/from_dict"
```

---

### Task 3: Engine trace hook

**Files:**
- Modify: `locma/core/engine.py:60-121` (`run_game`)
- Test: `tests/test_engine_on_step.py`

**Interfaces:**
- Consumes: existing `run_game(policy0, policy1, seed, cards=None, max_turns=200)`.
- Produces: `run_game(..., on_step=None)` where `on_step` is `Callable[[int, int | Action, GameState], None]` called after each applied policy decision. `seat` is the acting player (`gs.current` captured before apply). Draft picks pass the `int` index; battle steps pass the `Action`. Forced safety end-turns are NOT reported. Returns unchanged `GameResult`.

- [ ] **Step 1: Write the failing test**

```python
from locma.core.actions import Summon, Attack, Use, Pass
from locma.core.engine import run_game
from locma.policies.greedy import GreedyPolicy


def test_on_step_receives_draft_ints_then_battle_actions():
    steps = []
    run_game(GreedyPolicy(), GreedyPolicy(), seed=1, on_step=lambda s, a, gs: steps.append((s, a)))
    assert steps, "on_step should have been called"
    # draft picks are ints, battle actions are Action instances
    assert any(isinstance(a, int) for _, a in steps)
    assert any(isinstance(a, (Summon, Attack, Use, Pass)) for _, a in steps)
    # seats are always 0 or 1
    assert all(s in (0, 1) for s, _ in steps)


def test_on_step_none_is_default_and_harmless():
    r = run_game(GreedyPolicy(), GreedyPolicy(), seed=1)
    assert r.winner in (0, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine_on_step.py -v`
Expected: FAIL with `TypeError: run_game() got an unexpected keyword argument 'on_step'`.

- [ ] **Step 3: Modify `run_game` signature and loops**

Change the signature line:

```python
def run_game(policy0, policy1, seed: int, cards=None, max_turns: int = 200, on_step=None) -> GameResult:
```

In the draft loop, capture seat and report after apply:

```python
    while gs.phase == Phase.DRAFT:
        seat = gs.current
        view = make_draft_view(gs)
        pick = pols[gs.current].draft_action(view, [0, 1, 2])
        draftmod.apply_draft_pick(gs, pick)
        if on_step is not None:
            on_step(seat, pick, gs)
```

In the battle inner loop, capture seat and report after apply (leave the `per_turn > 100` forced `end_turn` unreported):

```python
        while gs.current == turn_owner and gs.phase == Phase.BATTLE:
            seat = gs.current
            legal = battlemod.battle_legal(gs)
            view = make_battle_view(gs)
            action = pols[gs.current].battle_action(view, legal)
            battlemod.apply_battle(gs, action)
            if on_step is not None:
                on_step(seat, action, gs)
            per_turn += 1
            if per_turn > 100:
                battlemod.end_turn(gs)
                break
```

- [ ] **Step 4: Run the new test plus the full suite**

Run: `uv run pytest tests/test_engine_on_step.py -v && uv run pytest -q`
Expected: new tests PASS; full suite still green (existing engine tests unaffected because `on_step` defaults to `None`).

- [ ] **Step 5: Commit**

```bash
git add locma/core/engine.py tests/test_engine_on_step.py
git commit -m "feat(core): optional on_step trace hook in run_game"
```

---

### Task 4: Trace recorder, canonical hash, game-log I/O

**Files:**
- Create: `locma/harness/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: `run_game` (Task 3), `action_to_dict`/`action_from_dict` (Task 2), `GameResult`.
- Produces:
  - `class Recorder` with `trace: list[tuple[int, int | Action]]` and method `record(seat, action, gs) -> None`.
  - `record_game(policy0, policy1, seed, cards=None) -> tuple[GameResult, list]` — runs a recorded game, returns `(result, recorder.trace)`.
  - `serialize_trace(trace) -> list[list]` — each entry `[seat, step_dict]`; draft picks encode as `{"t":"draft","pick":int}`, battle actions via `action_to_dict`.
  - `trace_hash(trace, winner, turns) -> str` — `"sha256:" + hexdigest` of `canonical_json(serialize_trace(trace) + [winner, turns])`.
  - `canonical_json(obj) -> str` — `json.dumps(obj, sort_keys=True, separators=(",", ":"))`.
  - `write_game_log(path, records: list[dict]) -> None` — append one JSON object per line.
  - `read_game_log(path) -> list[dict]` — parse JSONL into dicts.

- [ ] **Step 1: Write the failing test**

```python
import json

from locma.harness.trace import (
    Recorder, record_game, serialize_trace, trace_hash, canonical_json,
    write_game_log, read_game_log,
)
from locma.policies.greedy import GreedyPolicy


def test_record_game_returns_trace():
    result, trace = record_game(GreedyPolicy(), GreedyPolicy(), seed=3)
    assert result.winner in (0, 1)
    assert len(trace) > 0
    seat, action = trace[0]
    assert seat in (0, 1)


def test_hash_is_deterministic():
    r1, t1 = record_game(GreedyPolicy(), GreedyPolicy(), seed=7)
    r2, t2 = record_game(GreedyPolicy(), GreedyPolicy(), seed=7)
    assert trace_hash(t1, r1.winner, r1.turns) == trace_hash(t2, r2.winner, r2.turns)


def test_hash_changes_with_outcome():
    r, t = record_game(GreedyPolicy(), GreedyPolicy(), seed=7)
    h = trace_hash(t, r.winner, r.turns)
    assert h != trace_hash(t, 1 - r.winner, r.turns)


def test_canonical_json_is_sorted_compact():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_serialize_trace_encodes_draft_and_battle():
    _, trace = record_game(GreedyPolicy(), GreedyPolicy(), seed=1)
    ser = serialize_trace(trace)
    tags = {entry[1]["t"] for entry in ser}
    assert "draft" in tags


def test_game_log_roundtrip(tmp_path):
    path = tmp_path / "g.jsonl"
    rec = {"format": 1, "seed": 1, "winner": 0, "turns": 5, "hash": "sha256:abc"}
    write_game_log(str(path), [rec])
    assert read_game_log(str(path)) == [rec]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'locma.harness.trace'`.

- [ ] **Step 3: Implement `locma/harness/trace.py`**

```python
from __future__ import annotations

import hashlib
import json

from locma.core.actions import Action, action_to_dict
from locma.core.engine import GameResult, run_game


class Recorder:
    """Collects (seat, action) steps via the engine's on_step hook."""

    def __init__(self) -> None:
        self.trace: list[tuple[int, int | Action]] = []

    def record(self, seat: int, action: int | Action, gs) -> None:
        self.trace.append((seat, action))


def record_game(policy0, policy1, seed: int, cards=None) -> tuple[GameResult, list]:
    rec = Recorder()
    result = run_game(policy0, policy1, seed, cards=cards, on_step=rec.record)
    return result, rec.trace


def _encode_step(action: int | Action) -> dict:
    if isinstance(action, int):
        return {"t": "draft", "pick": action}
    return action_to_dict(action)


def serialize_trace(trace: list) -> list[list]:
    return [[seat, _encode_step(action)] for seat, action in trace]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def trace_hash(trace: list, winner: int, turns: int) -> str:
    payload = canonical_json(serialize_trace(trace) + [winner, turns])
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_game_log(path: str, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_game_log(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_trace.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add locma/harness/trace.py tests/test_trace.py
git commit -m "feat(harness): trace recorder, canonical hash, game-log I/O"
```

---

### Task 5: openskill ratings

**Files:**
- Create: `locma/stats/openskill_ratings.py`
- Modify: `pyproject.toml` (add `openskill` to `[project.dependencies]`)
- Test: `tests/test_openskill_ratings.py`

**Interfaces:**
- Consumes: `openskill.models.PlackettLuce`. Input `pairs: list[tuple[str, str, float]]` of `(a_name, b_name, score_a)` with `score_a` in `{0.0, 0.5, 1.0}` — identical shape to `elo_from_results`.
- Produces: `openskill_from_results(pairs) -> dict[str, tuple[float, float]]` mapping name → `(mu, sigma)`; and `ordinal(mu, sigma) -> float` returning `mu - 3 * sigma`.

- [ ] **Step 1: Write the failing test**

```python
from locma.stats.openskill_ratings import openskill_from_results, ordinal


def test_dominant_player_outranks_weak_one():
    # 'strong' beats 'weak' 20 times (score_a = 1.0 means a won)
    pairs = [("strong", "weak", 1.0) for _ in range(20)]
    ratings = openskill_from_results(pairs)
    s_mu, s_sigma = ratings["strong"]
    w_mu, w_sigma = ratings["weak"]
    assert ordinal(s_mu, s_sigma) > ordinal(w_mu, w_sigma)


def test_returns_mu_sigma_tuples():
    ratings = openskill_from_results([("a", "b", 1.0)])
    assert set(ratings) == {"a", "b"}
    for mu, sigma in ratings.values():
        assert isinstance(mu, float) and isinstance(sigma, float)


def test_ordinal_formula():
    assert ordinal(25.0, 8.0) == 25.0 - 3 * 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openskill_ratings.py -v`
Expected: FAIL (`ModuleNotFoundError` for `locma.stats.openskill_ratings`, and possibly for `openskill` until Step 4).

- [ ] **Step 3: Implement `locma/stats/openskill_ratings.py`**

```python
from __future__ import annotations

from openskill.models import PlackettLuce


def ordinal(mu: float, sigma: float) -> float:
    """Conservative skill estimate: mu - 3*sigma."""
    return mu - 3 * sigma


def openskill_from_results(pairs: list[tuple[str, str, float]]) -> dict[str, tuple[float, float]]:
    """Compute openskill (PlackettLuce) ratings from (a, b, score_a) results.

    score_a == 1.0 -> a won, 0.0 -> b won, 0.5 -> draw.
    Returns name -> (mu, sigma).
    """
    model = PlackettLuce()
    ratings: dict[str, object] = {}

    def get(name: str):
        if name not in ratings:
            ratings[name] = model.rating(name=name)
        return ratings[name]

    for a, b, score_a in pairs:
        ra, rb = get(a), get(b)
        if score_a == 0.5:
            ranks = [0, 0]
        elif score_a >= 1.0:
            ranks = [0, 1]  # a first (winner)
        else:
            ranks = [1, 0]  # b first
        [[ra2], [rb2]] = model.rate([[ra], [rb]], ranks=ranks)
        ratings[a], ratings[b] = ra2, rb2

    return {name: (float(r.mu), float(r.sigma)) for name, r in ratings.items()}
```

- [ ] **Step 4: Add `openskill` to core deps and sync**

In `pyproject.toml` change the dependencies line to:

```toml
dependencies = ["typer>=0.12", "rich>=13", "scipy>=1.11", "numpy>=1.26", "openskill>=5"]
```

Run: `uv sync --extra dev`
Expected: installs `openskill`.

> If the `model.rate(..., ranks=...)` call errors on the installed openskill version, consult current openskill docs (via context7 `resolve-library-id` → `query-docs` for "openskill") and adjust the rate call; the (mu, sigma) return contract stays the same.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_openskill_ratings.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock locma/stats/openskill_ratings.py tests/test_openskill_ratings.py
git commit -m "feat(stats): openskill ratings alongside Elo"
```

---

### Task 6: Live turn-by-turn renderer

**Files:**
- Create: `locma/cli/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `GameState` attributes (`gs.turn`, `gs.phase`, `gs.current`, `gs.players[i].health`, `.mana`, `.board`, `.hand`), `gs.opponent(i)`; `Phase` from `core/state`; action types from `core/actions`. rich `Console`.
- Produces: `class GameRenderer(console=None)` with `on_step(seat, action, gs) -> None` suitable as the `run_game` `on_step` callback. Prints a compact line per step (phase, turn, seat, action summary, both healths). Pure output; no return value.

- [ ] **Step 1: Write the failing test**

```python
from rich.console import Console

from locma.cli.render import GameRenderer
from locma.core.engine import run_game
from locma.policies.greedy import GreedyPolicy


def test_renderer_prints_without_error():
    console = Console(record=True, width=100)
    r = GameRenderer(console=console)
    run_game(GreedyPolicy(), GreedyPolicy(), seed=2, on_step=r.on_step)
    text = console.export_text()
    assert "turn" in text.lower() or "draft" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'locma.cli.render'`.

- [ ] **Step 3: Implement `locma/cli/render.py`**

```python
from __future__ import annotations

from rich.console import Console

from locma.core.actions import Attack, Pass, Summon, Use


def _action_summary(action) -> str:
    if isinstance(action, int):
        return f"draft pick {action}"
    if isinstance(action, Summon):
        return f"summon #{action.card_instance_id}"
    if isinstance(action, Attack):
        tgt = "face" if action.target_id == -1 else f"#{action.target_id}"
        return f"attack #{action.attacker_id} -> {tgt}"
    if isinstance(action, Use):
        tgt = "face/none" if action.target_id == -1 else f"#{action.target_id}"
        return f"use #{action.item_instance_id} -> {tgt}"
    if isinstance(action, Pass):
        return "pass"
    return repr(action)


class GameRenderer:
    """Prints a compact line per applied step, driven by run_game's on_step."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def on_step(self, seat: int, action, gs) -> None:
        if isinstance(action, int):
            self.console.print(f"[dim]draft[/] P{seat}: {_action_summary(action)}")
            return
        h0 = gs.players[0].health
        h1 = gs.players[1].health
        self.console.print(
            f"[bold]turn {gs.turn}[/] P{seat}: {_action_summary(action)}  "
            f"[green]hp {h0}[/] / [red]hp {h1}[/]"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_render.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add locma/cli/render.py tests/test_render.py
git commit -m "feat(cli): live turn-by-turn game renderer"
```

---

### Task 7: CLI rewire — sprt, noise-floor, replay, play render/log, tournament matrix+openskill

**Files:**
- Modify: `locma/cli/app.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `make_policy` (existing in `app.py`), `run_match`, `run_tournament`, `wilson_ci`, `binomial_test`, `sprt`, `record_game`/`trace_hash`/`serialize_trace`/`write_game_log`/`read_game_log` (Task 4), `openskill_from_results`/`ordinal` (Task 5), `GameRenderer` (Task 6), `elo_from_results` (existing).
- Produces final command set: `play` (`--render`, `--log`), `tournament` (`--matrix`, openskill column), `noise-floor`, `sprt` (replaces `eval`), `replay` (`--assert-hash`, `--render`), `fetch-cards`, `fetch-art`. Test via `typer.testing.CliRunner`.

> **Note on `--render`/`--log` for `play`:** these record per game. When either flag is set, `play` runs games one-by-one via `record_game` (so it can render/log), otherwise it uses the fast `run_match` path. Keep both paths; do not slow the default path.

- [ ] **Step 1: Write the failing test**

```python
from typer.testing import CliRunner

from locma.cli.app import app

runner = CliRunner()


def test_play_smoke():
    res = runner.invoke(app, ["play", "greedy", "random", "--games", "2", "--seed", "0"])
    assert res.exit_code == 0
    assert "win rate" in res.stdout.lower()


def test_sprt_smoke():
    res = runner.invoke(app, ["sprt", "greedy", "--vs", "random", "--max-games", "40", "--batch", "10"])
    assert res.exit_code == 0
    assert "verdict" in res.stdout.lower()


def test_eval_command_is_gone():
    res = runner.invoke(app, ["eval", "greedy"])
    assert res.exit_code != 0


def test_noise_floor_smoke():
    res = runner.invoke(app, ["noise-floor", "random", "--games", "20", "--seed", "0"])
    assert res.exit_code == 0
    assert "resolution limit" in res.stdout.lower()


def test_tournament_matrix_smoke():
    res = runner.invoke(app, ["tournament", "random", "greedy", "--games", "3", "--matrix"])
    assert res.exit_code == 0
    assert "openskill" in res.stdout.lower()


def test_play_log_then_replay_asserts_hash(tmp_path):
    log = tmp_path / "g.jsonl"
    r1 = runner.invoke(app, ["play", "greedy", "random", "--games", "2", "--seed", "5", "--log", str(log)])
    assert r1.exit_code == 0
    r2 = runner.invoke(app, ["replay", str(log), "--assert-hash"])
    assert r2.exit_code == 0
    assert "ok" in r2.stdout.lower()


def test_replay_detects_tampered_hash(tmp_path):
    import json
    log = tmp_path / "g.jsonl"
    runner.invoke(app, ["play", "greedy", "random", "--games", "1", "--seed", "5", "--log", str(log)])
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    rows[0]["hash"] = "sha256:deadbeef"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = runner.invoke(app, ["replay", str(log), "--assert-hash"])
    assert res.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (e.g. `noise-floor`/`sprt`/`replay` not found; `eval` still exists).

- [ ] **Step 3: Rewrite `locma/cli/app.py`**

Replace the entire file with:

```python
from __future__ import annotations

import importlib.metadata

import typer
from rich.console import Console
from rich.table import Table

from locma.cli.render import GameRenderer
from locma.harness.match import run_match
from locma.harness.tournament import run_tournament
from locma.harness.trace import (
    read_game_log,
    record_game,
    serialize_trace,
    trace_hash,
    write_game_log,
)
from locma.policies.greedy import GreedyPolicy
from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy
from locma.stats.intervals import binomial_test, wilson_ci
from locma.stats.openskill_ratings import openskill_from_results, ordinal
from locma.stats.sprt import sprt as sprt_test

app = typer.Typer(help="Legends of Code & Magic 1.2 explore kit")
console = Console()


def _version() -> str:
    try:
        return importlib.metadata.version("locma")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def make_policy(spec: str):
    table = {"random": RandomPolicy, "scripted": ScriptedPolicy, "greedy": GreedyPolicy}
    if spec in table:
        return table[spec](spec)
    raise typer.BadParameter(f"unknown policy '{spec}'")


@app.command()
def play(
    a: str,
    b: str,
    games: int = 100,
    seed: int = 0,
    render: bool = typer.Option(False, help="render each game turn-by-turn as played"),
    log: str = typer.Option(None, help="write a game-log JSONL (enables replay)"),
):
    """Run a mirrored match A vs B; optionally render and/or log each game."""
    pa, pb = make_policy(a), make_policy(b)
    wins_a = total = 0
    if render or log:
        records: list[dict] = []
        renderer = GameRenderer(console) if render else None
        for k in range(games):
            s = seed + k
            # mirrored pair: game1 A=seat0, game2 A=seat1
            for a_seat in (0, 1):
                p0, p1 = (pa, pb) if a_seat == 0 else (pb, pa)
                if renderer:
                    console.rule(f"game seed={s} a_seat={a_seat}")
                    res, trace = _recorded_with_render(p0, p1, s, renderer)
                else:
                    res, trace = record_game(p0, p1, seed=s)
                won = (res.winner == 0) if a_seat == 0 else (res.winner == 1)
                wins_a += int(won)
                total += 1
                if log:
                    records.append(
                        {
                            "format": 1,
                            "engine_version": _version(),
                            "policy_a": a,
                            "policy_b": b,
                            "seed": s,
                            "a_seat": a_seat,
                            "actions": serialize_trace(trace),
                            "winner": res.winner,
                            "turns": res.turns,
                            "hash": trace_hash(trace, res.winner, res.turns),
                        }
                    )
        if log:
            write_game_log(log, records)
    else:
        res = run_match(pa, pb, games=games, seed=seed)
        wins_a, total = res.wins_a, res.games

    lo, hi = wilson_ci(wins_a, total)
    p = binomial_test(wins_a, total, 0.5)
    console.print(
        f"[bold]{a}[/] vs [bold]{b}[/]  win rate A = {wins_a / total:.3f} "
        f"(95% CI {lo:.3f}-{hi:.3f}), p={p:.4g}, n={total}"
    )


def _recorded_with_render(p0, p1, seed, renderer):
    """Run a game, rendering each step and recording the trace."""
    from locma.core.engine import run_game  # noqa: PLC0415 — local to keep import graph flat
    from locma.harness.trace import Recorder  # noqa: PLC0415

    rec = Recorder()

    def on_step(seat, action, gs):
        rec.record(seat, action, gs)
        renderer.on_step(seat, action, gs)

    result = run_game(p0, p1, seed, on_step=on_step)
    return result, rec.trace


@app.command()
def tournament(
    names: list[str],
    games: int = 50,
    seed: int = 0,
    reference: str = "random",
    matrix: bool = typer.Option(False, help="print the pair-score matrix"),
):
    """Round-robin tournament with openskill (primary) and Elo ratings."""
    pols = [make_policy(n) for n in names]
    res = run_tournament(pols, games=games, seed=seed, reference=reference)

    # openskill from the same win matrix (reconstruct per-game results from win rates)
    pairs: list[tuple[str, str, float]] = []
    seen: set[frozenset[str]] = set()
    for (x, y), rate in res.win_matrix.items():
        key = frozenset((x, y))
        if key in seen:
            continue
        seen.add(key)
        wins_x = round(rate * games * 2)
        for _ in range(wins_x):
            pairs.append((x, y, 1.0))
        for _ in range(games * 2 - wins_x):
            pairs.append((x, y, 0.0))
    osk = openskill_from_results(pairs)

    t = Table(title="Ratings")
    t.add_column("policy")
    t.add_column("openskill", justify="right")
    t.add_column("elo", justify="right")
    t.add_column("p vs ref", justify="right")
    order = sorted(res.ratings, key=lambda k: -ordinal(*osk.get(k, (25.0, 8.333))))
    for n in order:
        mu, sigma = osk.get(n, (25.0, 8.333))
        t.add_row(
            n,
            f"{ordinal(mu, sigma):.2f}",
            f"{res.ratings[n]:.0f}",
            f"{res.p_vs_reference.get(n, float('nan')):.4g}",
        )
    console.print(t)

    if matrix:
        m = Table(title="Pair-score matrix (row win rate vs column)")
        m.add_column("")
        for n in names:
            m.add_column(n, justify="right")
        for row in names:
            cells = [row]
            for col in names:
                if row == col:
                    cells.append("--")
                else:
                    cells.append(f"{res.win_matrix.get((row, col), float('nan')):.2f}")
            m.add_row(*cells)
        console.print(m)


@app.command("noise-floor")
def noise_floor(a: str, games: int = 200, seed: int = 0):
    """Play policy A against an independent copy of itself: the luck baseline."""
    res = run_match(make_policy(a), make_policy(a), games=games, seed=seed)
    lo, hi = wilson_ci(res.wins_a, res.games)
    half = (hi - lo) / 2
    console.print(
        f"[bold]{a}[/] vs itself  win rate = {res.win_rate_a:.3f} "
        f"(95% CI {lo:.3f}-{hi:.3f}), n={res.games}\n"
        f"resolution limit: +/-{half:.3f}  "
        f"[dim](edges smaller than this are indistinguishable from luck)[/]"
    )


@app.command()
def sprt(
    x: str,
    vs: str = "random",
    p0: float = 0.5,
    p1: float = 0.6,
    max_games: int = 1000,
    batch: int = 20,
    seed: int = 0,
):
    """Sequential probability ratio test; stops as soon as evidence decides."""
    px, py = make_policy(x), make_policy(vs)
    wins = n = k = 0
    r = None
    while n < max_games:
        res = run_match(px, py, games=batch, seed=seed + k)
        k += batch
        wins += res.wins_a
        n += res.games
        r = sprt_test(wins, n, p0, p1)
        if r.decision != "continue":
            break
    lo, hi = wilson_ci(wins, n)
    console.print(
        f"verdict: [bold]{r.decision}[/]  winrate={wins / n:.3f} (CI {lo:.3f}-{hi:.3f}), n={n}"
    )


@app.command()
def replay(
    file: str,
    assert_hash: bool = typer.Option(False, "--assert-hash", help="fail if recomputed hash differs"),
    render: bool = typer.Option(False, help="render each replayed game"),
):
    """Re-simulate a logged game and (optionally) assert byte-identical hash."""
    rows = read_game_log(file)
    mismatches = 0
    for i, row in enumerate(rows):
        pa, pb = make_policy(row["policy_a"]), make_policy(row["policy_b"])
        p0, p1 = (pa, pb) if row["a_seat"] == 0 else (pb, pa)
        if render:
            renderer = GameRenderer(console)
            console.rule(f"replay game {i} seed={row['seed']}")
            result, trace = _recorded_with_render(p0, p1, row["seed"], renderer)
        else:
            result, trace = record_game(p0, p1, seed=row["seed"])
        h = trace_hash(trace, result.winner, result.turns)
        ok = h == row.get("hash")
        if not ok:
            mismatches += 1
            console.print(f"[red]game {i}: hash MISMATCH[/] stored={row.get('hash')} got={h}")
        else:
            console.print(f"game {i}: ok ({h})")
    if assert_hash and mismatches:
        raise typer.Exit(code=1)


@app.command("fetch-cards")
def fetch_cards_cmd():
    from locma.data.fetch import fetch_cards  # noqa: PLC0415 — lazy import

    path = fetch_cards()
    console.print(f"cards at {path}")


@app.command("fetch-art")
def fetch_art_cmd():
    from locma.data.fetch import fetch_art  # noqa: PLC0415 — lazy import

    n = fetch_art()
    console.print(f"fetched {n} art assets (best-effort)")
```

> **Note — `sprt` naming:** the command function is named `sprt`, so the statistic is imported as `sprt_test` (`from locma.stats.sprt import sprt as sprt_test`) and called as `sprt_test(wins, n, p0, p1)` to avoid shadowing. This is already reflected in the code above.

- [ ] **Step 4: Run the CLI tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all 7 PASS.

- [ ] **Step 5: Lint and full suite**

Run: `uv run ruff check locma tests && uv run ruff format locma tests && uv run pytest -q`
Expected: `All checks passed!` and full suite green.

- [ ] **Step 6: Commit**

```bash
git add locma/cli/app.py tests/test_cli.py
git commit -m "feat(cli): sprt/noise-floor/replay commands, play render+log, tournament matrix+openskill"
```

---

### Task 8: Docs split

**Files:**
- Create: `docs/cli.md`, `docs/experiments.md`, `docs/architecture.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the final command set and formats from Tasks 1–7.
- Produces: documentation only. No code symbols. Verification is structural (files exist, README links resolve, examples match real flags).

- [ ] **Step 1: Create `docs/cli.md`**

Full command reference. Document every command with its flags and a runnable example, copied to match Task 7 exactly:

```markdown
# CLI Reference

All commands run via `uv run locma <command>`.

## play
`locma play A B [--games N] [--seed S] [--render] [--log FILE]`
Run a mirrored match. Prints win rate + 95% Wilson CI + binomial p-value.
- `--render` renders each game turn-by-turn as played.
- `--log FILE` writes a game-log JSONL (one record per game) enabling `replay`.

Example: `uv run locma play greedy random --games 50 --seed 0 --log run.jsonl`

## tournament
`locma tournament A B C... [--games N] [--seed S] [--reference R] [--matrix]`
Round-robin. Ratings table: policy | openskill (ordinal) | elo | p vs reference.
- `--matrix` prints the pair-score matrix (row win rate vs column).

Example: `uv run locma tournament random scripted greedy --games 30 --matrix`

## noise-floor
`locma noise-floor A [--games N] [--seed S]`
Plays A against an independent copy of itself — the luck baseline. Prints win
rate, CI, and the resolution limit (CI half-width).

Example: `uv run locma noise-floor greedy --games 500`

## sprt
`locma sprt X --vs B [--p0 0.5] [--p1 0.6] [--max-games N] [--batch K] [--seed S]`
Sequential probability ratio test; stops as soon as evidence decides. Prints
verdict (accept_h1 / accept_h0 / continue), win rate, CI, and n.

Example: `uv run locma sprt greedy --vs random --max-games 200`

## replay
`locma replay FILE [--assert-hash] [--render]`
Re-simulates each logged game from its seed and recomputes the content hash.
- `--assert-hash` exits non-zero on any mismatch.
- `--render` shows each replayed game.

Example: `uv run locma replay run.jsonl --assert-hash`

## fetch-cards / fetch-art
Refresh the vendored card list / best-effort art download.
```

- [ ] **Step 2: Create `docs/experiments.md`**

```markdown
# Experiment Methodology

## Noise floor (luck baseline)
`noise-floor` plays a policy against an independent copy of itself. It answers:
"how big must a win-rate edge be before it's real?"
- **Stochastic policies** (e.g. random): win rate centers on 0.50; the CI width
  is the measurement floor.
- **Deterministic policies** (greedy/scripted): self-play variance comes only
  from seat asymmetry and the seed's RNG draws, so the win rate may sit stably
  off 0.50. Read the **resolution limit** (CI half-width), not the point value:
  any measured edge smaller than it is indistinguishable from luck.

## SPRT (sequential testing)
`sprt` tests H0: winrate = p0 against H1: winrate = p1 using Wald's
log-likelihood ratio, batching games until the LLR crosses an acceptance
boundary (alpha = beta = 0.05) or `--max-games` is hit. It stops as soon as the
evidence decides — far fewer games than a fixed-n test for clear effects.

## Ratings: Elo and openskill
`tournament` reports both. Elo is the classic pairwise update; openskill
(Plackett-Luce) tracks a (mu, sigma) belief and reports a conservative
**ordinal** = mu - 3*sigma. openskill is the primary number (it models
uncertainty); Elo is kept for continuity and comparison.

## Replay & determinism
Every game is a deterministic function of (seed, policies). `play --log` records
the action sequence and a content hash = sha256(canonical_json(actions +
[winner, turns])). `replay --assert-hash` re-runs from the seed and fails if the
recomputed hash differs — catching any accidental change to engine or policy
behavior.
```

- [ ] **Step 3: Create `docs/architecture.md`**

```markdown
# Architecture

## Layering
Pure rules engine at the center; policies, Gym/SB3, stats, and CLI are layers
around it. ML deps stay behind the `[ml]` extra and never enter core.

## Trace hook
`run_game(..., on_step=None)` calls `on_step(seat, action, gs)` after each
applied policy decision (draft pick int, or battle Action). Default `None` means
zero overhead for tournaments/sprt/noise-floor. Two consumers:
- `harness/trace.Recorder` collects `(seat, action)` pairs.
- `cli/render.GameRenderer` prints turn-by-turn.

## Game-log format
One JSON object per line (a match -> a JSONL file):
`{format, engine_version, policy_a, policy_b, seed, a_seat, actions, winner,
turns, hash}`. `actions` is the serialized trace; `hash` is
`sha256:<hexdigest>` of `canonical_json(actions + [winner, turns])`.

## Determinism guarantee
`policy.reset(seed)` + `random.Random(seed)` make each game reproducible
regardless of play order, which is what makes byte-identical replay possible.
```

- [ ] **Step 4: Trim `README.md` to an intro + links**

Replace the detailed "CLI commands" block in `README.md` with a short pointer, keeping Install and the intro:

```markdown
## Documentation

- [CLI reference](docs/cli.md) — every command and flag
- [Experiment methodology](docs/experiments.md) — noise floor, SPRT, ratings, replay
- [Architecture](docs/architecture.md) — engine, trace hook, game-log format

## Quickstart

```bash
uv sync                      # core install
uv sync --extra ml           # + Gym env & SB3 training
uv run locma play greedy random --games 50 --seed 0
uv run locma tournament random scripted greedy --games 30 --matrix
uv run pytest
```
```

- [ ] **Step 5: Verify docs and examples**

Run: `uv run locma --help && uv run locma play --help && uv run locma replay --help`
Expected: help text matches the flags documented in `docs/cli.md` (cross-check `--render`, `--log`, `--matrix`, `--assert-hash` all appear).

- [ ] **Step 6: Commit**

```bash
git add README.md docs/cli.md docs/experiments.md docs/architecture.md
git commit -m "docs: split CLI/experiments/architecture into docs/, trim README"
```

---

## Final verification

- [ ] Run full suite: `uv run pytest -q` → all tests pass (51 existing + new).
- [ ] Lint clean: `uv run ruff check .` → `All checks passed!`.
- [ ] Format clean: `uv run ruff format --check .` → no files would be reformatted.
- [ ] Manual smoke: `uv run locma play greedy random --games 3 --log /tmp/g.jsonl && uv run locma replay /tmp/g.jsonl --assert-hash` → replay reports `ok` and exits 0.
