# locma/server/app.py
from __future__ import annotations

import glob
import os
import random as _random
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from locma import depot as dep
from locma.data.cards_db import catalog, load_cards
from locma.harness.interactive import IllegalMove, InteractiveGame, WrongPhase
from locma.harness.replay_store import get_replay, list_headers, write_replay
from locma.harness.replay_stream import build_replay, build_replay_from_log_row
from locma.harness.trace import read_game_log
from locma.policies.registry import make_policy, policy_names
from locma.server.depot_api import depot_router
from locma.server.experiments import experiments_router
from locma.server.jobs import JobRunner
from locma.server.session_store import SessionStore


class RunRequest(BaseModel):
    policy_a: str
    policy_b: str
    seed: int = 0


class ImportRequest(BaseModel):
    path: str
    row: int = 0


class NewGameRequest(BaseModel):
    opponent: str
    seed: int | None = None


class DraftRequest(BaseModel):
    pick: int


class ActionRequest(BaseModel):
    action: dict


def create_app(
    replay_dir: str,
    asset_dir: str,
    gamelog_dir: str,
    presets_dir: str = "experiments/presets",
    results_dir: str = "runs/experiments",
    workers: int = 0,
) -> FastAPI:
    runner = JobRunner(results_dir=results_dir, workers=workers)

    @asynccontextmanager
    async def _lifespan(_app):
        yield
        runner.shutdown()

    app = FastAPI(title="locma panel server", lifespan=_lifespan)
    cards = catalog()  # static; load once
    engine_cards = load_cards()  # Card objects for the engine (catalog() is JSON for the UI)
    sessions = SessionStore()
    app.include_router(experiments_router(runner, presets_dir))
    app.include_router(depot_router())

    def _make_policy(spec: str):
        """make_policy with web-friendly errors (bad spec / missing [ml] extra)."""
        try:
            return make_policy(spec)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ImportError as e:
            raise HTTPException(
                status_code=400, detail=f"'{spec}' requires the [ml] extra: {e}"
            ) from e

    def _persist_if_finished(game: InteractiveGame) -> None:
        if game.result is not None and not getattr(game, "_persisted", False):
            write_replay(replay_dir, game._replay)
            game._persisted = True

    @app.get("/api/cards")
    def get_cards() -> list[dict]:
        return cards

    @app.get("/api/policies")
    def get_policies() -> list[str]:
        return policy_names()

    @app.get("/api/policy-catalog")
    def policy_catalog() -> dict:
        """Everything the UI needs to build a policy spec: baseline names,
        model-backed spec templates, and the depot's pinned model artifacts."""
        models = []
        for name in dep.artifact_names():
            rec = dep.load_record(name)
            if rec["kind"] != "model" or not rec["pin"]:
                continue
            vrec = next(v for v in rec["versions"] if v["version"] == rec["pin"])
            models.append(
                {
                    "name": name,
                    "version": rec["pin"],
                    "refs": [f"depot:{name}/{f}" for f in sorted(vrec["files"])],
                }
            )
        suggestions = list(policy_names())
        suggestions += ["mcts:100", "dmcts:15,30", "azlite:100"]
        for m in models:
            for ref in m["refs"]:
                suggestions += [f"ppo:{ref}", f"vbeam:{ref}"]
        return {
            "baselines": policy_names(),
            "model_bases": [
                {"base": "ppo", "template": "ppo:MODEL"},
                {"base": "vbeam", "template": "vbeam:MODEL,8,20"},
                {"base": "netdmcts", "template": "netdmcts:15,80,1.5,MODEL"},
            ],
            "depot_models": models,
            "suggestions": suggestions,
        }

    @app.post("/api/replays")
    def run_replay(req: RunRequest) -> dict:
        p_a = _make_policy(req.policy_a)
        p_b = _make_policy(req.policy_b)
        rep = build_replay(p_a, p_b, req.seed, source="ad-hoc")
        write_replay(replay_dir, rep)
        return rep["header"]

    @app.get("/api/replays")
    def get_index() -> list[dict]:
        return list_headers(replay_dir)

    @app.get("/api/replays/{rid}")
    def get_one(rid: str) -> dict:
        try:
            return get_replay(replay_dir, rid)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail="replay not found") from e

    @app.get("/api/game-logs")
    def list_game_logs() -> list[dict]:
        out = []
        for p in sorted(glob.glob(os.path.join(gamelog_dir, "*.jsonl"))):
            try:
                rows = len(read_game_log(p))
            except OSError:
                continue
            out.append({"path": os.path.basename(p), "rows": rows})
        return out

    @app.post("/api/replays/import")
    def import_replay(req: ImportRequest) -> dict:
        full = os.path.join(gamelog_dir, os.path.basename(req.path))
        try:
            rows = read_game_log(full)
            row = rows[req.row]
        except (OSError, IndexError) as e:
            raise HTTPException(status_code=400, detail="bad game-log or row") from e
        source = f"game-log:{os.path.basename(req.path)}#{req.row}"
        try:
            rep = build_replay_from_log_row(row, source=source, make_policy=make_policy)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        write_replay(replay_dir, rep)
        return rep["header"]

    @app.get("/api/art-index")
    def art_index() -> list[int]:
        """Card ids with cached portraits — the client consults this once and
        never requests missing art (no 404 spam when the cache is empty;
        populate it with `locma fetch-art`)."""
        if not os.path.isdir(asset_dir):
            return []
        ids = []
        for fname in os.listdir(asset_dir):
            stem, ext = os.path.splitext(fname)
            if ext == ".png" and stem.isdigit():
                ids.append(int(stem))
        return sorted(ids)

    @app.get("/api/art/{card_id}")
    def get_art(card_id: int):
        path = os.path.join(asset_dir, f"{card_id:03d}.png")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="art not found")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/games")
    def new_game(req: NewGameRequest) -> dict:
        ai = _make_policy(req.opponent)
        seed = req.seed if req.seed is not None else _random.randint(0, 2**31 - 1)
        game = sessions.create(ai_policy=ai, seed=seed, cards=engine_cards, rng=_random)
        _persist_if_finished(game)
        return {
            "game_id": game.game_id,
            "you": game.human_seat,
            "status": game.status,
            "pending": game.pending(),
            "result": game.result,
        }

    @app.get("/api/games/{gid}")
    def get_game(gid: str) -> dict:
        game = sessions.get(gid)
        if game is None:
            raise HTTPException(status_code=404, detail="game not found")
        return {"status": game.status, "pending": game.pending(), "result": game.result}

    @app.post("/api/games/{gid}/draft")
    def game_draft(gid: str, req: DraftRequest) -> dict:
        game = sessions.get(gid)
        if game is None:
            raise HTTPException(status_code=404, detail="game not found")
        try:
            resp = game.submit_draft(req.pick)
        except WrongPhase as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except IllegalMove as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        _persist_if_finished(game)
        return resp

    @app.post("/api/games/{gid}/action")
    def game_action(gid: str, req: ActionRequest) -> dict:
        game = sessions.get(gid)
        if game is None:
            raise HTTPException(status_code=404, detail="game not found")
        try:
            resp = game.submit_action(req.action)
        except WrongPhase as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except (IllegalMove, KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        _persist_if_finished(game)
        return resp

    dist = os.path.join(os.path.dirname(__file__), "..", "..", "web", "dist")
    if os.path.isdir(dist):
        app.mount("/", StaticFiles(directory=dist, html=True), name="spa")

    app.state.replay_dir = replay_dir
    app.state.asset_dir = asset_dir
    app.state.gamelog_dir = gamelog_dir
    app.state.make_policy = make_policy
    return app
