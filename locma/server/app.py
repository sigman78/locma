# locma/server/app.py
from __future__ import annotations

from fastapi import FastAPI

from locma.data.cards_db import catalog
from locma.policies.registry import make_policy

POLICIES = ["random", "scripted", "greedy"]


def create_app(replay_dir: str, asset_dir: str, gamelog_dir: str) -> FastAPI:
    app = FastAPI(title="locma replay server")
    cards = catalog()  # static; load once

    @app.get("/api/cards")
    def get_cards() -> list[dict]:
        return cards

    @app.get("/api/policies")
    def get_policies() -> list[str]:
        return POLICIES

    # later tasks register replay / game-log / art / static routes on `app`
    app.state.replay_dir = replay_dir
    app.state.asset_dir = asset_dir
    app.state.gamelog_dir = gamelog_dir
    app.state.make_policy = make_policy
    return app
