"""Experiment kinds, presets, and the /api/experiments router for the web panel.

An experiment *kind* bundles a param schema (drives the UI form), a ``plan``
(params + job dir -> a Plan of picklable cells, optionally a metrics tail for
long cells) and a ``reduce`` (cell results -> result dict). A kind may also
provide a ``stream`` factory: a closure called on the collector thread as
each cell finishes, appending live points to ``job.series`` / partial state
to ``job.live`` for the panel's charts. Adding a new experiment to the panel
= adding one ``_Kind`` entry here; the UI renders forms from the schema and
charts from the series — no frontend changes needed.

Cell functions must stay top-level (they cross process boundaries on
``workers > 1``, exactly like the ceiling-eval pool). Policy parameters are
registry specs and accept ``depot:`` refs (``vbeam:depot:b0/b0_s0.zip``).

Presets are one JSON file per preset in ``presets_dir`` (committed —
shareable, diff-able): ``{"name", "kind", "params", "note"}``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from locma.envs.training import ZOO_OPPONENTS  # constant, no [ml] needed
from locma.server.jobs import Job, JobRunner, TailConfig, make_job_id
from locma.stats.intervals import binomial_test, wilson_ci

_MATCH_CHUNK = 10  # game-pairs per cell: progress granularity vs dispatch overhead


# ---------------------------------------------------------------------------
# cell functions (top-level: picklable units of work)
# ---------------------------------------------------------------------------


def match_cell(spec_a: str, spec_b: str, games: int, seed: int) -> dict:
    """One chunk of a mirrored match; identical seed layout to ``locma play``."""
    from locma.harness.match import run_match  # noqa: PLC0415
    from locma.policies.registry import make_policy  # noqa: PLC0415

    res = run_match(make_policy(spec_a), make_policy(spec_b), games=games, seed=seed)
    return {"wins_a": res.wins_a, "games": res.games}


def ceiling_cell(spec: str, seed: int, games_per_seed: int, opponents: tuple[str, ...]) -> float:
    from locma.harness.ceiling_eval import avg_hard3_at_seed  # noqa: PLC0415

    return avg_hard3_at_seed(spec, seed, games_per_seed, opponents)


def train_cell(p: dict, out_path: str, metrics_path: str, cancel_path: str) -> dict:
    """One full train-zoo run. Streams metrics as JSONL (tailed by the runner
    for live charts + progress) and polls ``cancel_path`` between steps —
    touching it makes SB3 stop cleanly; the model is saved either way."""
    from stable_baselines3.common.callbacks import BaseCallback  # noqa: PLC0415

    from locma.envs.training import train_zoo  # noqa: PLC0415

    class _MetricsCallback(BaseCallback):
        def _on_rollout_end(self) -> None:
            buf = self.model.ep_info_buffer
            rec: dict = {"timesteps": int(self.num_timesteps)}
            if buf:
                rec["ep_rew_mean"] = float(sum(e["r"] for e in buf) / len(buf))
            # latest logged train/* diagnostics (previous update; fine for curves)
            for k, v in self.model.logger.name_to_value.items():
                if isinstance(v, int | float):
                    rec[k.rpartition("/")[2]] = float(v)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        def _on_step(self) -> bool:
            return not os.path.exists(cancel_path)

    saved = train_zoo(
        opponents=p["opponents"],
        steps_per_opponent=p["steps_per_opponent"],
        out=out_path,
        seed=p["seed"],
        ent_coef=p["ent_coef"],
        verbose=0,
        obs_mode=p["obs_mode"],
        learning_rate=p["learning_rate"],
        target_kl=p["target_kl"] or None,
        n_steps=p["n_steps"],
        n_envs=p["n_envs"],
        draft_noise=p["draft_noise"],
        callback=_MetricsCallback(),
    )
    return {"out": saved, "cancelled": os.path.exists(cancel_path)}


# ---------------------------------------------------------------------------
# kinds
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    cells: list
    tail: TailConfig | None = None  # for long cells that stream metrics to a file
    cancel_file: str | None = None  # cooperative cancel inside a running cell


@dataclass
class _Kind:
    label: str
    description: str
    schema: list[dict]  # [{name, type: policy|policies|int|float|str, default, help?}]
    plan: Any  # (params, job_dir) -> Plan
    reduce: Any  # (params, results) -> result dict
    stream: Any = None  # params -> on_cell(job, index, result) closure, or None


def _chunks(total: int, size: int) -> list[tuple[int, int]]:
    """(offset, count) chunks covering ``total``."""
    return [(off, min(size, total - off)) for off in range(0, total, size)]


# -- match / noise-floor ------------------------------------------------------


def _plan_match(p: dict, job_dir: str) -> Plan:
    return Plan(
        [
            (match_cell, (p["policy_a"], p["policy_b"], count, p["seed"] + off))
            for off, count in _chunks(p["games"], _MATCH_CHUNK)
        ]
    )


def _reduce_match(p: dict, results: list) -> dict:
    wins = sum(r["wins_a"] for r in results)
    games = sum(r["games"] for r in results)
    lo, hi = wilson_ci(wins, games)
    return {
        "policy_a": p["policy_a"],
        "policy_b": p["policy_b"],
        "wins_a": wins,
        "games": games,
        "win_rate": wins / games if games else 0.0,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value": binomial_test(wins, games, 0.5),
    }


def _stream_match(p: dict):
    """Running win rate + Wilson CI band vs games played (chunks may land out
    of order; the accumulator is order-independent)."""
    state = {"wins": 0, "games": 0}

    def on_cell(job: Job, i: int, result: dict) -> None:
        state["wins"] += result["wins_a"]
        state["games"] += result["games"]
        lo, hi = wilson_ci(state["wins"], state["games"])
        x = state["games"]
        job.add_point("win_rate", x, state["wins"] / state["games"])
        job.add_point("ci_lo", x, lo)
        job.add_point("ci_hi", x, hi)

    return on_cell


def _plan_noise_floor(p: dict, job_dir: str) -> Plan:
    return _plan_match(dict(p, policy_a=p["policy"], policy_b=p["policy"]), job_dir)


def _reduce_noise_floor(p: dict, results: list) -> dict:
    out = _reduce_match(dict(p, policy_a=p["policy"], policy_b=p["policy"]), results)
    out["resolution"] = (out["ci_hi"] - out["ci_lo"]) / 2
    return out


# -- league --------------------------------------------------------------------


def _league_pairs(p: dict) -> list[tuple[str, str]]:
    from itertools import combinations  # noqa: PLC0415

    names = p["policies"]
    if len(names) != len(set(names)):
        raise ValueError("league policies must be unique")
    return list(combinations(names, 2))


def _plan_league(p: dict, job_dir: str) -> Plan:
    return Plan([(match_cell, (a, b, p["games"], p["seed"])) for a, b in _league_pairs(p)])


def _reduce_league(p: dict, results: list) -> dict:
    from locma.stats.openskill_ratings import openskill_from_results, ordinal  # noqa: PLC0415
    from locma.stats.ratings import elo_from_results  # noqa: PLC0415

    names = p["policies"]
    reference = p.get("reference") or None
    matrix: dict[str, dict[str, float]] = {a: {} for a in names}
    game_pairs: list[tuple[str, str, float]] = []
    vs_ref: dict[str, list[int]] = {n: [0, 0] for n in names}

    for (a, b), r in zip(_league_pairs(p), results, strict=True):
        wins_a, games = r["wins_a"], r["games"]
        matrix[a][b] = wins_a / games if games else 0.0
        matrix[b][a] = 1.0 - matrix[a][b]
        game_pairs += [(a, b, 1.0)] * wins_a + [(a, b, 0.0)] * (games - wins_a)
        if reference in (a, b):
            other, w = (b, games - wins_a) if a == reference else (a, wins_a)
            vs_ref[other][0] += w
            vs_ref[other][1] += games

    elo = elo_from_results(game_pairs)
    osk = openskill_from_results(game_pairs)
    table = []
    for n in names:
        mu, sigma = osk.get(n, (25.0, 8.333))
        w, g = vs_ref[n]
        table.append(
            {
                "policy": n,
                "openskill": ordinal(mu, sigma),
                "elo": elo.get(n, 1500.0),
                "p_vs_ref": binomial_test(w, g, 0.5) if g and n != reference else None,
                "avg_win_rate": sum(matrix[n].values()) / max(1, len(matrix[n])),
            }
        )
    table.sort(key=lambda row: -row["openskill"])
    return {"table": table, "matrix": matrix, "policies": names, "reference": reference}


def _stream_league(p: dict):
    """The pair matrix fills in cell by cell (job.live, not a line series)."""
    pairs = _league_pairs(p)

    def on_cell(job: Job, i: int, result: dict) -> None:
        a, b = pairs[i]
        wr = result["wins_a"] / result["games"] if result["games"] else 0.0
        matrix = job.live.setdefault("matrix", {})
        matrix.setdefault(a, {})[b] = wr
        matrix.setdefault(b, {})[a] = 1.0 - wr
        job.live["policies"] = p["policies"]

    return on_cell


# -- ceiling ---------------------------------------------------------------------


def _plan_ceiling(p: dict, job_dir: str) -> Plan:
    from locma.harness.ceiling_eval import _disjoint_eval_seeds  # noqa: PLC0415

    seeds = _disjoint_eval_seeds(p["seeds"], p["games_per_seed"])
    opponents = tuple(p["opponents"])
    paths = list(p["candidates"]) + list(p["baselines"])
    return Plan(
        [(ceiling_cell, (path, s, p["games_per_seed"], opponents)) for path in paths for s in seeds]
    )


def _reduce_ceiling(p: dict, results: list) -> dict:
    import numpy as np  # noqa: PLC0415

    from locma.harness.ceiling_eval import decide, paired_bootstrap_ci  # noqa: PLC0415

    n_seeds = p["seeds"]
    cand_n = len(p["candidates"])
    rows = np.array(results, dtype=float).reshape(cand_n + len(p["baselines"]), n_seeds)
    cand = rows[:cand_n].mean(axis=0)  # per-seed mean over candidate models
    base = rows[cand_n:].mean(axis=0)
    mean_delta, lo, hi = paired_bootstrap_ci(cand - base)
    return {
        "cand_avg": float(cand.mean()),
        "b0_avg": float(base.mean()),
        "mean_delta": mean_delta,
        "ci_lo": lo,
        "ci_hi": hi,
        "verdict": decide(mean_delta, lo, hi, p["threshold"]),
        "candidates": p["candidates"],
        "baselines": p["baselines"],
        "opponents": p["opponents"],
    }


def _stream_ceiling(p: dict):
    """Per-seed paired deltas as both arms complete a seed, plus the running
    mean — the verdict visibly converging."""
    n_seeds = p["seeds"]
    n_cand = len(p["candidates"])
    n_paths = n_cand + len(p["baselines"])
    got: dict[tuple[int, int], float] = {}
    deltas: dict[int, float] = {}

    def on_cell(job: Job, i: int, result: float) -> None:
        got[(i // n_seeds, i % n_seeds)] = float(result)
        s = i % n_seeds
        col = [got.get((path, s)) for path in range(n_paths)]
        if any(v is None for v in col):
            return
        cand = sum(col[:n_cand]) / n_cand
        base = sum(col[n_cand:]) / (n_paths - n_cand)
        deltas[s] = cand - base
        job.add_point("delta", s, deltas[s])
        job.add_point("mean_delta", len(deltas), sum(deltas.values()) / len(deltas))

    return on_cell


# -- train-zoo ---------------------------------------------------------------------


def _plan_train(p: dict, job_dir: str) -> Plan:
    out_path = os.path.join(job_dir, "model.zip")
    metrics_path = os.path.join(job_dir, "metrics.jsonl")
    cancel_path = os.path.join(job_dir, "cancel")
    total = p["steps_per_opponent"] * len(p["opponents"])
    return Plan(
        cells=[(train_cell, (p, out_path, metrics_path, cancel_path))],
        tail=TailConfig(path=metrics_path, x="timesteps", total=total),
        cancel_file=cancel_path,
    )


def _reduce_train(p: dict, results: list) -> dict:
    out = dict(results[0])
    out["total_timesteps"] = p["steps_per_opponent"] * len(p["opponents"])
    out["opponents"] = p["opponents"]
    return out


KINDS: dict[str, _Kind] = {
    "match": _Kind(
        label="Head-to-head match",
        description="Mirrored A vs B; win rate + 95% Wilson CI + binomial p (locma play).",
        schema=[
            {"name": "policy_a", "type": "policy", "default": "greedy"},
            {"name": "policy_b", "type": "policy", "default": "random"},
            {"name": "games", "type": "int", "default": 100, "help": "mirrored game pairs"},
            {"name": "seed", "type": "int", "default": 0},
        ],
        plan=_plan_match,
        reduce=_reduce_match,
        stream=_stream_match,
    ),
    "noise-floor": _Kind(
        label="Noise floor",
        description="A vs an independent copy of itself: the luck baseline and resolution limit.",
        schema=[
            {"name": "policy", "type": "policy", "default": "greedy"},
            {"name": "games", "type": "int", "default": 200, "help": "mirrored game pairs"},
            {"name": "seed", "type": "int", "default": 0},
        ],
        plan=_plan_noise_floor,
        reduce=_reduce_noise_floor,
        stream=_stream_match,
    ),
    "league": _Kind(
        label="League (round-robin)",
        description="Round-robin over a policy list; openskill + Elo table and pair matrix "
        "(locma tournament).",
        schema=[
            {
                "name": "policies",
                "type": "policies",
                "default": ["random", "scripted", "greedy", "max-guard", "max-attack"],
            },
            {"name": "games", "type": "int", "default": 50, "help": "game pairs per pairing"},
            {"name": "seed", "type": "int", "default": 0},
            {"name": "reference", "type": "policy", "default": "random", "help": "for p-values"},
        ],
        plan=_plan_league,
        reduce=_reduce_league,
        stream=_stream_league,
    ),
    "ceiling": _Kind(
        label="Paired verdict (ceiling-eval)",
        description="Paired per-seed avg win-rate delta of candidates minus baselines over "
        "held-out seeds, bootstrap CI + verdict (locma ceiling-eval).",
        schema=[
            {"name": "candidates", "type": "policies", "default": ["vbeam:depot:b0/b0_s0.zip"]},
            {"name": "baselines", "type": "policies", "default": ["depot:b0/b0_s0.zip"]},
            {"name": "seeds", "type": "int", "default": 10, "help": "held-out eval seeds"},
            {"name": "games_per_seed", "type": "int", "default": 10},
            {
                "name": "opponents",
                "type": "policies",
                "default": ["scripted", "max-guard", "max-attack"],
            },
            {"name": "threshold", "type": "float", "default": 0.03},
        ],
        plan=_plan_ceiling,
        reduce=_reduce_ceiling,
        stream=_stream_ceiling,
    ),
    "train-zoo": _Kind(
        label="Train (zoo curriculum)",
        description="Train one MaskablePPO net back-to-back against the opponent list "
        "(locma train-zoo). Live reward/loss curves; the checkpoint lands in the job "
        "directory and can be published to the depot. Requires the [ml] extra; "
        "one training job at a time.",
        schema=[
            {"name": "opponents", "type": "policies", "default": list(ZOO_OPPONENTS)},
            {"name": "steps_per_opponent", "type": "int", "default": 200_000},
            {"name": "seed", "type": "int", "default": 0},
            {"name": "obs_mode", "type": "str", "default": "token", "help": "token or flat"},
            {"name": "learning_rate", "type": "float", "default": 1e-4},
            {"name": "target_kl", "type": "float", "default": 0.025, "help": "0 = off"},
            {"name": "ent_coef", "type": "float", "default": 0.02},
            {"name": "n_steps", "type": "int", "default": 2048, "help": "rollout length"},
            {"name": "n_envs", "type": "int", "default": 1, "help": "parallel envs"},
            {"name": "draft_noise", "type": "int", "default": 0},
        ],
        plan=_plan_train,
        reduce=_reduce_train,
    ),
}


def _coerce_params(kind: _Kind, params: dict) -> dict:
    """Fill defaults and coerce field types; unknown keys are rejected."""
    known = {f["name"] for f in kind.schema}
    unknown = set(params) - known
    if unknown:
        raise ValueError(f"unknown params: {sorted(unknown)}")
    out: dict = {}
    for f in kind.schema:
        v = params.get(f["name"], f["default"])
        t = f["type"]
        try:
            if t == "int":
                v = int(v)
            elif t == "float":
                v = float(v)
            elif t == "policy":
                v = str(v)
            elif t == "policies":
                if isinstance(v, str):
                    v = [s.strip() for s in v.split(",") if s.strip()]
                v = [str(s) for s in v]
                if not v:
                    raise ValueError("empty list")
            else:
                v = str(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad value for '{f['name']}': {e}") from e
        out[f["name"]] = v
    return out


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

_PRESET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class PresetBody(BaseModel):
    name: str
    kind: str
    params: dict
    note: str = ""


class RunBody(BaseModel):
    kind: str
    params: dict
    name: str = ""


def _preset_path(presets_dir: str, pid: str) -> str:
    if not _PRESET_ID_RE.match(pid):
        raise HTTPException(status_code=400, detail=f"bad preset id '{pid}'")
    return os.path.join(presets_dir, f"{pid}.json")


def experiments_router(runner: JobRunner, presets_dir: str) -> APIRouter:
    router = APIRouter(prefix="/api/experiments")

    @router.get("/kinds")
    def kinds() -> list[dict]:
        return [
            {"kind": k, "label": v.label, "description": v.description, "schema": v.schema}
            for k, v in KINDS.items()
        ]

    @router.get("/presets")
    def list_presets() -> list[dict]:
        out = []
        if os.path.isdir(presets_dir):
            for fname in sorted(os.listdir(presets_dir)):
                if not fname.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(presets_dir, fname), encoding="utf-8") as f:
                        body = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                body["id"] = fname[: -len(".json")]
                out.append(body)
        return out

    @router.put("/presets/{pid}")
    def save_preset(pid: str, body: PresetBody) -> dict:
        if body.kind not in KINDS:
            raise HTTPException(status_code=400, detail=f"unknown kind '{body.kind}'")
        try:
            params = _coerce_params(KINDS[body.kind], body.params)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        path = _preset_path(presets_dir, pid)
        os.makedirs(presets_dir, exist_ok=True)
        doc = {"name": body.name, "kind": body.kind, "params": params, "note": body.note}
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        return dict(doc, id=pid)

    @router.delete("/presets/{pid}")
    def delete_preset(pid: str) -> dict:
        path = _preset_path(presets_dir, pid)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="preset not found")
        os.remove(path)
        return {"deleted": pid}

    @router.post("/run")
    def run(body: RunBody) -> dict:
        kind = KINDS.get(body.kind)
        if kind is None:
            raise HTTPException(status_code=400, detail=f"unknown kind '{body.kind}'")
        job_id = make_job_id(body.kind)
        try:
            params = _coerce_params(kind, body.params)
            plan: Plan = kind.plan(params, runner.job_dir(job_id))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        job = runner.submit(
            kind=body.kind,
            name=body.name or kind.label,
            params=params,
            cells=plan.cells,
            reduce_fn=kind.reduce,
            on_cell=kind.stream(params) if kind.stream else None,
            tail=plan.tail,
            cancel_file=plan.cancel_file,
            job_id=job_id,
        )
        return job.to_dict(include_series=False)

    @router.get("/jobs")
    def jobs() -> list[dict]:
        live = [j.to_dict(include_series=False) for j in runner.list()]
        return sorted(live + runner.history(), key=lambda j: j["created"], reverse=True)

    @router.get("/jobs/{job_id}")
    def job(job_id: str) -> dict:
        j = runner.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="job not found")
        return j.to_dict(include_series=False)

    @router.get("/jobs/{job_id}/series")
    def series(job_id: str) -> dict:
        j = runner.get(job_id)
        if j is not None:
            return {"series": j.series, "live": j.live}
        doc = runner.persisted(job_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"series": doc.get("series", {}), "live": doc.get("live", {})}

    @router.get("/jobs/{job_id}/log")
    def log(job_id: str) -> dict:
        if runner.get(job_id) is None and runner.persisted(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"log": runner.log_text(job_id) or ""}

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        if not runner.cancel(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        return {"cancelled": job_id}

    return router
