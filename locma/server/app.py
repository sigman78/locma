# locma/server/app.py
from __future__ import annotations

import glob
import os
import random as _random

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from locma.data.cards_db import catalog, load_cards
from locma.harness.interactive import IllegalMove, InteractiveGame, WrongPhase
from locma.harness.replay_store import get_replay, list_headers, write_replay
from locma.harness.replay_stream import build_replay, build_replay_from_log_row
from locma.harness.trace import read_game_log
from locma.policies.registry import make_policy, policy_names
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


def create_app(replay_dir: str, asset_dir: str, gamelog_dir: str) -> FastAPI:
    app = FastAPI(title="locma replay server")
    cards = catalog()  # static; load once
    engine_cards = load_cards()  # Card objects for the engine (catalog() is JSON for the UI)
    sessions = SessionStore()

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

    @app.post("/api/replays")
    def run_replay(req: RunRequest) -> dict:
        try:
            p_a = make_policy(req.policy_a)
            p_b = make_policy(req.policy_b)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
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

    @app.get("/api/art/{card_id}")
    def get_art(card_id: int):
        path = os.path.join(asset_dir, f"{card_id:03d}.png")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="art not found")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/games")
    def new_game(req: NewGameRequest) -> dict:
        try:
            ai = make_policy(req.opponent)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
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
