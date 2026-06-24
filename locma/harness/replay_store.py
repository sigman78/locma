from __future__ import annotations

import glob
import json
import os
from itertools import groupby


def _full_path(dirpath: str, rid: str) -> str:
    return os.path.join(dirpath, f"{rid}.jsonl")


def write_replay(dirpath: str, replay: dict) -> str:
    os.makedirs(dirpath, exist_ok=True)
    rid = replay["header"]["replay_id"]
    path = _full_path(dirpath, rid)

    draft = replay["draft"]
    battle = replay["battle"]

    lines: list[str] = []

    # line 1: meta
    lines.append(json.dumps({"k": "meta", "header": replay["header"]}, separators=(",", ":")))

    # draft lines — one per round; skipped entirely when pool is None
    pool = draft.get("pool")
    if pool is not None:
        picks_by_round: dict[int, list[dict]] = {}
        for p in draft.get("picks", []):
            picks_by_round.setdefault(p["round"], []).append(p)
        for r, round_pool in enumerate(pool):
            round_picks = [
                {"seat": p["seat"], "pick": p["pick"]} for p in picks_by_round.get(r, [])
            ]
            lines.append(
                json.dumps(
                    {"k": "draft", "round": r, "pool": round_pool, "picks": round_picks},
                    separators=(",", ":"),
                )
            )

    # opening snapshot
    opening = battle.get("opening")
    if opening is not None:
        lines.append(json.dumps({"k": "open", "state": opening}, separators=(",", ":")))

    # battle turn lines — group by CONSECUTIVE (seat, turn) runs
    steps = battle.get("steps", [])
    for (seat, turn), group in groupby(steps, key=lambda s: (s["seat"], s["turn"])):
        actions = [{"action": s["action"], "state": s["state"]} for s in group]
        lines.append(
            json.dumps(
                {"k": "turn", "seat": seat, "turn": turn, "actions": actions},
                separators=(",", ":"),
            )
        )

    # result line
    lines.append(json.dumps({"k": "result", **replay["result"]}, separators=(",", ":")))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return path


def get_replay(dirpath: str, replay_id: str) -> dict:
    path = _full_path(dirpath, replay_id)
    if not os.path.exists(path):
        raise FileNotFoundError(replay_id)

    header: dict | None = None
    pool: list | None = None
    picks: list[dict] = []
    opening: dict | None = None
    steps: list[dict] = []
    result: dict = {}
    has_draft_lines = False

    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            k = line.get("k")
            if k == "meta":
                header = line["header"]
            elif k == "draft":
                has_draft_lines = True
                r = line["round"]
                if pool is None:
                    pool = []
                # extend pool list to cover this round index
                while len(pool) <= r:
                    pool.append([])
                pool[r] = line["pool"]
                for p in line.get("picks", []):
                    picks.append({"round": r, "seat": p["seat"], "pick": p["pick"]})
            elif k == "open":
                opening = line["state"]
            elif k == "turn":
                seat = line["seat"]
                turn = line["turn"]
                for entry in line.get("actions", []):
                    steps.append(
                        {
                            "seat": seat,
                            "turn": turn,
                            "action": entry["action"],
                            "state": entry["state"],
                        }
                    )
            elif k == "result":
                result = {kk: v for kk, v in line.items() if kk != "k"}

    # If no draft lines were seen, pool stays None and picks stays []
    if not has_draft_lines:
        pool = None

    return {
        "header": header,
        "draft": {"pool": pool, "picks": picks},
        "battle": {"opening": opening, "steps": steps},
        "result": result,
    }


def list_headers(dirpath: str) -> list[dict]:
    heads: list[dict] = []
    for p in glob.glob(os.path.join(dirpath, "*.jsonl")):
        try:
            with open(p, encoding="utf-8") as f:
                first = f.readline()
            if not first.strip():
                continue
            line = json.loads(first)
            if line.get("k") != "meta":
                continue
            heads.append(line["header"])
        except (OSError, json.JSONDecodeError):
            continue
    heads.sort(key=lambda h: h.get("created_at", ""), reverse=True)
    return heads
