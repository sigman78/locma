"""Experiment kinds, presets, and the /api/experiments router for the web panel.

An experiment *kind* bundles a param schema (drives the UI form), a ``plan``
(params -> picklable cells for the job runner) and a ``reduce`` (cell results
-> result dict). Adding a new experiment to the panel = adding one ``_Kind``
entry here; the UI renders it from the schema, presets save its params, and
the job runner parallelizes it — no frontend changes needed.

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

from locma.server.jobs import JobRunner
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


# ---------------------------------------------------------------------------
# kinds
# ---------------------------------------------------------------------------


@dataclass
class _Kind:
    label: str
    description: str
    schema: list[dict]  # [{name, type: policy|policies|int|float|str, default, help?}]
    plan: Any  # params -> list[Cell]
    reduce: Any  # (params, results) -> result dict


def _chunks(total: int, size: int) -> list[tuple[int, int]]:
    """(offset, count) chunks covering ``total``."""
    return [(off, min(size, total - off)) for off in range(0, total, size)]


def _plan_match(p: dict) -> list:
    return [
        (match_cell, (p["policy_a"], p["policy_b"], count, p["seed"] + off))
        for off, count in _chunks(p["games"], _MATCH_CHUNK)
    ]


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


def _plan_noise_floor(p: dict) -> list:
    q = dict(p, policy_a=p["policy"], policy_b=p["policy"])
    return _plan_match(q)


def _reduce_noise_floor(p: dict, results: list) -> dict:
    out = _reduce_match(dict(p, policy_a=p["policy"], policy_b=p["policy"]), results)
    out["resolution"] = (out["ci_hi"] - out["ci_lo"]) / 2
    return out


def _league_pairs(p: dict) -> list[tuple[str, str]]:
    from itertools import combinations  # noqa: PLC0415

    names = p["policies"]
    if len(names) != len(set(names)):
        raise ValueError("league policies must be unique")
    return list(combinations(names, 2))


def _plan_league(p: dict) -> list:
    return [(match_cell, (a, b, p["games"], p["seed"])) for a, b in _league_pairs(p)]


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


def _plan_ceiling(p: dict) -> list:
    from locma.harness.ceiling_eval import _disjoint_eval_seeds  # noqa: PLC0415

    seeds = _disjoint_eval_seeds(p["seeds"], p["games_per_seed"])
    opponents = tuple(p["opponents"])
    paths = list(p["candidates"]) + list(p["baselines"])
    return [
        (ceiling_cell, (path, s, p["games_per_seed"], opponents)) for path in paths for s in seeds
    ]


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
        try:
            params = _coerce_params(kind, body.params)
            cells = kind.plan(params)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        job = runner.submit(
            kind=body.kind,
            name=body.name or kind.label,
            params=params,
            cells=cells,
            reduce_fn=kind.reduce,
        )
        return job.to_dict()

    @router.get("/jobs")
    def jobs() -> list[dict]:
        live = [j.to_dict() for j in runner.list()]
        return sorted(live + runner.history(), key=lambda j: j["created"], reverse=True)

    @router.get("/jobs/{job_id}")
    def job(job_id: str) -> dict:
        j = runner.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="job not found")
        return j.to_dict()

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        if not runner.cancel(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        return {"cancelled": job_id}

    return router
