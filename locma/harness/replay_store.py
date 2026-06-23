from __future__ import annotations

import glob
import json
import os


def _full_path(dirpath: str, rid: str) -> str:
    return os.path.join(dirpath, f"{rid}.json")


def _meta_path(dirpath: str, rid: str) -> str:
    return os.path.join(dirpath, f"{rid}.meta.json")


def write_replay(dirpath: str, replay: dict) -> str:
    os.makedirs(dirpath, exist_ok=True)
    rid = replay["header"]["replay_id"]
    full = _full_path(dirpath, rid)
    with open(full, "w", encoding="utf-8") as f:
        json.dump(replay, f, separators=(",", ":"))
    with open(_meta_path(dirpath, rid), "w", encoding="utf-8") as f:
        json.dump(replay["header"], f, separators=(",", ":"))
    return full


def list_headers(dirpath: str) -> list[dict]:
    heads: list[dict] = []
    for p in glob.glob(os.path.join(dirpath, "*.meta.json")):
        try:
            with open(p, encoding="utf-8") as f:
                heads.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    heads.sort(key=lambda h: h.get("created_at", ""), reverse=True)
    return heads


def get_replay(dirpath: str, replay_id: str) -> dict:
    full = _full_path(dirpath, replay_id)
    if not os.path.exists(full):
        raise FileNotFoundError(replay_id)
    with open(full, encoding="utf-8") as f:
        return json.load(f)
