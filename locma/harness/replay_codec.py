# locma/harness/replay_codec.py
from __future__ import annotations

import hashlib
from functools import lru_cache

from locma.data.cards_db import load_cards

# Player scalar fields carried in a state snapshot.
SCALARS = ("health", "mana", "max_mana", "damage_counter", "bonus_draw", "deck_count")


@lru_cache(maxsize=1)
def _catalog() -> dict:
    return {c.id: c for c in load_cards()}


def cardlist_version() -> str:
    """Short stable digest of the printed-card stats used as the compression dictionary."""
    payload = ";".join(
        f"{cid}:{c.attack}:{c.defense}:{c.abilities}" for cid, c in sorted(_catalog().items())
    )
    return "cl_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def compact_card(card: dict, *, board: bool) -> list:
    """Full card dict -> [iid, card_id] or [iid, card_id, dev].

    `dev` carries only fields that deviate from the printed card (atk/def/abilities),
    plus board readiness when not the (can_attack=True, has_attacked=False) default.
    """
    cid = card["card_id"]
    base = _catalog().get(cid)
    dev: dict = {}
    if base is None:
        dev["atk"] = card["atk"]
        dev["def"] = card["def"]
        dev["abilities"] = card["abilities"]
    else:
        if card["atk"] != base.attack:
            dev["atk"] = card["atk"]
        if card["def"] != base.defense:
            dev["def"] = card["def"]
        if card["abilities"] != base.abilities:
            dev["abilities"] = card["abilities"]
    if board:
        if card["can_attack"] is not True:
            dev["can_attack"] = card["can_attack"]
        if card["has_attacked"] is not False:
            dev["has_attacked"] = card["has_attacked"]
    ref = [card["iid"], cid]
    if dev:
        ref.append(dev)
    return ref


def expand_card(ref: list, *, board: bool) -> dict:
    """[iid, card_id(, dev)] -> full card dict, filling base stats from the catalog."""
    iid, cid = ref[0], ref[1]
    dev = ref[2] if len(ref) > 2 else {}
    base = _catalog().get(cid)
    if base is None:
        atk, defense, abilities = dev["atk"], dev["def"], dev["abilities"]
    else:
        atk = dev.get("atk", base.attack)
        defense = dev.get("def", base.defense)
        abilities = dev.get("abilities", base.abilities)
    out = {"iid": iid, "card_id": cid, "atk": atk, "def": defense, "abilities": abilities}
    if board:
        out["can_attack"] = dev.get("can_attack", True)
        out["has_attacked"] = dev.get("has_attacked", False)
    return out


def _compact_zone(cards: list, *, board: bool) -> list:
    return [compact_card(c, board=board) for c in cards]


def _expand_zone(refs: list, *, board: bool) -> list:
    return [expand_card(r, board=board) for r in refs]


def compact_state(state: dict) -> dict:
    """Full state -> compact keyframe (all scalars + catalog-filled zones)."""
    players = []
    for p in state["players"]:
        op = {k: p[k] for k in SCALARS}
        op["hand"] = _compact_zone(p["hand"], board=False)
        op["board"] = _compact_zone(p["board"], board=True)
        players.append(op)
    return {"current": state["current"], "players": players}


def expand_state(compact: dict) -> dict:
    players = []
    for p in compact["players"]:
        op = {k: p[k] for k in SCALARS}
        op["hand"] = _expand_zone(p["hand"], board=False)
        op["board"] = _expand_zone(p["board"], board=True)
        players.append(op)
    return {"current": compact["current"], "players": players}


def diff_state(prev: dict, cur: dict) -> dict:
    """Delta of full state `cur` vs full state `prev`."""
    d: dict = {}
    if cur["current"] != prev["current"]:
        d["cur"] = cur["current"]
    seats = []
    for seat in (0, 1):
        pc, pp = cur["players"][seat], prev["players"][seat]
        entry: dict = {}
        s = {k: pc[k] for k in SCALARS if pc[k] != pp[k]}
        if s:
            entry["s"] = s
        if pc["hand"] != pp["hand"]:
            entry["hand"] = _compact_zone(pc["hand"], board=False)
        if pc["board"] != pp["board"]:
            entry["board"] = _compact_zone(pc["board"], board=True)
        if entry:
            entry["seat"] = seat
            seats.append(entry)
    if seats:
        d["p"] = seats
    return d


def apply_delta(running: dict, d: dict) -> None:
    """Apply delta `d` to full state `running` in place."""
    if "cur" in d:
        running["current"] = d["cur"]
    for entry in d.get("p", []):
        p = running["players"][entry["seat"]]
        for k, v in entry.get("s", {}).items():
            p[k] = v
        if "hand" in entry:
            p["hand"] = _expand_zone(entry["hand"], board=False)
        if "board" in entry:
            p["board"] = _expand_zone(entry["board"], board=True)
