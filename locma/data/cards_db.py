"""Card database loader for Legends of Code & Magic 1.2.

Vendored card list source:
  https://raw.githubusercontent.com/ronaldosvieira/gym-locm/master/gym_locm/engine/resources/cardlist.txt

The file is semicolon-delimited with 11 columns (all with surrounding spaces):
  id ; name ; type_word ; cost ; attack ; defense ; abilities ; player_hp ; enemy_hp ;
  card_draw ; description

Where type_word maps to CardType:
  creature  -> CardType.CREATURE  (0)
  itemGreen -> CardType.GREEN_ITEM (1)
  itemRed   -> CardType.RED_ITEM   (2)
  itemBlue  -> CardType.BLUE_ITEM  (3)

abilities is a 6-char mask in BCDGLW order (e.g. "-----W", "B-D-L-").
"""

from __future__ import annotations

import re
from importlib import resources

from locma.core.cards import Card, CardType, normalize_abilities

_TYPE_MAP: dict[str, CardType] = {
    "creature": CardType.CREATURE,
    "itemgreen": CardType.GREEN_ITEM,
    "itemred": CardType.RED_ITEM,
    "itemblue": CardType.BLUE_ITEM,
}


def parse_cardlist(text: str) -> list[Card]:
    """Parse a semicolon-delimited cardlist text into a list of Card objects.

    Handles the real LOCM 1.2 format:
      id ; name ; type_word ; cost ; attack ; defense ; abilities ; player_hp ; enemy_hp ;
      card_draw ; description
    """
    cards: list[Card] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 10:
            continue
        cid = int(parts[0])
        name = parts[1]
        ctype = _TYPE_MAP[parts[2].lower()]
        cost = int(parts[3])
        attack = int(parts[4])
        defense = int(parts[5])
        abilities = normalize_abilities(parts[6])
        player_hp = int(parts[7])
        enemy_hp = int(parts[8])
        card_draw = int(parts[9])
        # parts[10] is the description — not stored in Card
        cards.append(
            Card(cid, name, ctype, cost, attack, defense, abilities, player_hp, enemy_hp, card_draw)
        )
    return cards


def load_cards() -> list[Card]:
    """Load all 160 cards from the vendored cardlist.txt.

    Raises AssertionError if the file does not contain exactly 160 cards.
    """
    text = resources.files("locma.data").joinpath("cardlist.txt").read_text(encoding="utf-8")
    cards = parse_cardlist(text)
    if len(cards) != 160:
        raise ValueError(f"expected 160 cards, got {len(cards)}")
    return cards


def card_by_id(cards: list[Card]) -> dict[int, Card]:
    """Return a mapping from card id to Card."""
    return {c.id: c for c in cards}


_KEYWORD_NAMES = {"breakthrough", "charge", "drain", "guard", "lethal", "ward"}
_CREATURE_PREFACE = re.compile(r"^\s*\d+/\d+\s+creature\b[.\s]*", re.IGNORECASE)
_ITEM_PREFACE = re.compile(r"^\s*(?:green|red|blue)\s+item\b[\s.:;,–-]*", re.IGNORECASE)


def _sgn(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)


def _creature_special(description: str) -> str:
    """A creature's description reduced to its special on-summon/effect text: drops the
    "X/Y Creature." preface and any keyword-only sentences (e.g. "Charge, Drain.").
    Returns "" for a vanilla creature. The keyword check is per comma-separated token, so a
    real special that merely mentions a keyword (e.g. "Summon: give Lethal") is kept."""
    body = _CREATURE_PREFACE.sub("", description)
    kept: list[str] = []
    for sentence in body.split("."):
        s = sentence.strip()
        if not s:
            continue
        tokens = [t.strip() for t in s.split(",") if t.strip()]
        if tokens and all(t.lower() in _KEYWORD_NAMES for t in tokens):
            continue
        kept.append(s)
    return f"{'. '.join(kept)}." if kept else ""


def _item_effect(row: dict) -> str:
    """An item's effect text: its description with the "<Colour> item." preface removed,
    falling back to a derived stat/HP/draw summary when there is no description text."""
    cleaned = _ITEM_PREFACE.sub("", row["description"]).strip()
    if cleaned:
        return cleaned
    parts: list[str] = []
    if row["attack"] or row["defense"]:
        parts.append(f"{_sgn(row['attack'])}/{_sgn(row['defense'])}")
    if row["player_hp"]:
        parts.append(f"{_sgn(row['player_hp'])}♥")
    if row["enemy_hp"]:
        parts.append(f"foe {_sgn(row['enemy_hp'])}♥")
    if row["card_draw"]:
        parts.append(f"draw {_sgn(row['card_draw'])}")
    return " · ".join(parts)


def card_text(row: dict) -> str:
    """Cleaned special/effect text for a card dict: the creature special for creatures,
    the item effect for items (see ``_creature_special`` / ``_item_effect``)."""
    if row["type"] == "creature":
        return _creature_special(row["description"])
    return _item_effect(row)


def catalog() -> list[dict]:
    """Return all 160 cards as plain dicts including the raw ``description`` and the
    cleaned ``card_text`` (special/effect) column."""
    text = resources.files("locma.data").joinpath("cardlist.txt").read_text(encoding="utf-8")
    rows: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 10:
            continue
        row = {
            "id": int(parts[0]),
            "name": parts[1],
            "type": parts[2].lower(),
            "cost": int(parts[3]),
            "attack": int(parts[4]),
            "defense": int(parts[5]),
            "abilities": normalize_abilities(parts[6]),
            "player_hp": int(parts[7]),
            "enemy_hp": int(parts[8]),
            "card_draw": int(parts[9]),
            "description": parts[10] if len(parts) > 10 else "",
        }
        row["card_text"] = card_text(row)
        rows.append(row)
    return rows
