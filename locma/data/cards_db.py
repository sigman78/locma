"""Card database loader for Legends of Code & Magic 1.2.

Vendored card list source:
  https://raw.githubusercontent.com/ronaldosvieira/gym-locm/master/gym_locm/engine/resources/cardlist.txt

The file is semicolon-delimited with 11 columns (all with surrounding spaces):
  id ; name ; type_word ; cost ; attack ; defense ; abilities ; player_hp ; enemy_hp ; card_draw ; description

Where type_word maps to CardType:
  creature  -> CardType.CREATURE  (0)
  itemGreen -> CardType.GREEN_ITEM (1)
  itemRed   -> CardType.RED_ITEM   (2)
  itemBlue  -> CardType.BLUE_ITEM  (3)

abilities is a 6-char mask in BCDGLW order (e.g. "-----W", "B-D-L-").
"""
from __future__ import annotations

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
      id ; name ; type_word ; cost ; attack ; defense ; abilities ; player_hp ; enemy_hp ; card_draw ; description
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
    assert len(cards) == 160, f"expected 160 cards, got {len(cards)}"
    return cards


def card_by_id(cards: list[Card]) -> dict[int, Card]:
    """Return a mapping from card id to Card."""
    return {c.id: c for c in cards}
