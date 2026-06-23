# locma/server/app.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from locma.data.cards_db import catalog
from locma.harness.replay_store import get_replay, list_headers, write_replay
from locma.harness.replay_stream import build_replay
from locma.policies.registry import make_policy

POLICIES = ["random", "scripted", "greedy"]


class RunRequest(BaseModel):
    policy_a: str
    policy_b: str
    seed: int = 0


def create_app(replay_dir: str, asset_dir: str, gamelog_dir: str) -> FastAPI:
    app = FastAPI(title="locma replay server")
    cards = catalog()  # static; load once

    @app.get("/api/cards")
    def get_cards() -> list[dict]:
        return cards

    @app.get("/api/policies")
    def get_policies() -> list[str]:
        return POLICIES

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

    # later tasks register replay / game-log / art / static routes on `app`
    app.state.replay_dir = replay_dir
    app.state.asset_dir = asset_dir
    app.state.gamelog_dir = gamelog_dir
    app.state.make_policy = make_policy
    return app
