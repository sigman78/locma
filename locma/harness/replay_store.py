from __future__ import annotations

import glob
import json
import os
from itertools import groupby

from locma.harness.replay_codec import compact_state, diff_state


def _full_path(dirpath: str, rid: str) -> str:
    return os.path.join(dirpath, f"{rid}.jsonl")


def _j(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def write_replay(dirpath: str, replay: dict) -> str:
    os.makedirs(dirpath, exist_ok=True)
    rid = replay["header"]["replay_id"]
    path = _full_path(dirpath, rid)
    if replay["header"].get("format") == "locma-replay/3":
        lines = _encode_v3(replay)
    else:
        lines = _encode_v2(replay)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _encode_v2(replay: dict) -> list[str]:
    draft = replay["draft"]
    battle = replay["battle"]
    lines: list[str] = [_j({"k": "meta", "header": replay["header"]})]
    pool = draft.get("pool")
    if pool is not None:
        picks_by_round: dict[int, list[dict]] = {}
        for p in draft.get("picks", []):
            picks_by_round.setdefault(p["round"], []).append(p)
        for r, round_pool in enumerate(pool):
            round_picks = [
                {"seat": p["seat"], "pick": p["pick"]} for p in picks_by_round.get(r, [])
            ]
            lines.append(_j({"k": "draft", "round": r, "pool": round_pool, "picks": round_picks}))
    opening = battle.get("opening")
    if opening is not None:
        lines.append(_j({"k": "open", "state": opening}))
    for (seat, turn), group in groupby(
        battle.get("steps", []), key=lambda s: (s["seat"], s["turn"])
    ):
        actions = [
            {"action": s["action"], "state": s["state"], "events": s.get("events", [])}
            for s in group
        ]
        lines.append(_j({"k": "turn", "seat": seat, "turn": turn, "actions": actions}))
    closing = battle.get("closing")
    if closing is not None:
        lines.append(_j({"k": "close", "state": closing}))
    lines.append(_j({"k": "result", **replay["result"]}))
    return lines


def _encode_v3(replay: dict) -> list[str]:
    header = replay["header"]
    draft = replay["draft"]
    battle = replay["battle"]
    lines: list[str] = [_j({"k": "meta", "header": header})]
    pool = draft.get("pool")
    if pool is not None:
        lines.append(
            _j(
                {
                    "k": "draft_start",
                    "seed": header.get("seed"),
                    "source": header.get("source"),
                    "rounds": len(pool),
                }
            )
        )
        picks_by_round: dict[int, list[dict]] = {}
        for p in draft.get("picks", []):
            picks_by_round.setdefault(p["round"], []).append(p)
        for r, round_pool in enumerate(pool):
            round_picks = [
                {"seat": p["seat"], "pick": p["pick"]} for p in picks_by_round.get(r, [])
            ]
            lines.append(_j({"k": "draft", "round": r, "pool": round_pool, "picks": round_picks}))
        lines.append(_j({"k": "draft_end"}))
    opening = battle.get("opening")
    steps = battle.get("steps", [])
    if opening is None and steps:
        raise ValueError("locma-replay/3 requires an opening keyframe when steps are present")
    prev = opening
    if opening is not None:
        lines.append(_j({"k": "battle_start", "keyframe": compact_state(opening)}))
    for (seat, turn), group in groupby(steps, key=lambda s: (s["seat"], s["turn"])):
        actions = []
        for s in group:
            actions.append(
                {
                    "action": s["action"],
                    "d": diff_state(prev, s["state"]),
                    "events": s.get("events", []),
                }
            )
            prev = s["state"]
        lines.append(_j({"k": "turn", "seat": seat, "turn": turn, "actions": actions}))
    closing = battle.get("closing")
    if closing is not None and prev is not None:
        lines.append(_j({"k": "battle_end", "d": diff_state(prev, closing)}))
    lines.append(_j({"k": "result", **replay["result"]}))
    return lines


def get_replay(dirpath: str, replay_id: str) -> dict:
    path = _full_path(dirpath, replay_id)
    if not os.path.exists(path):
        raise FileNotFoundError(replay_id)

    header: dict | None = None
    pool: list | None = None
    picks: list[dict] = []
    opening: dict | None = None
    closing: dict | None = None
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
                pool.append(line["pool"])
                for p in line.get("picks", []):
                    picks.append({"round": r, "seat": p["seat"], "pick": p["pick"]})
            elif k == "open":
                opening = line["state"]
            elif k == "close":
                closing = line["state"]
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
                            "events": entry.get("events", []),
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
        "battle": {"opening": opening, "steps": steps, "closing": closing},
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
