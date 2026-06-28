from __future__ import annotations

from locma.core.cards import CardType
from locma.data.cards_db import card_by_id, card_text, catalog, load_cards, parse_cardlist


def test_load_cards_count():
    cards = load_cards()
    assert len(cards) == 160
    assert all(1 <= c.id <= 160 for c in cards)


def test_card_by_id_unique():
    cards = load_cards()
    by_id = card_by_id(cards)
    assert len(by_id) == 160


def test_parse_one_line():
    # Real line copied verbatim from locma/data/cardlist.txt (line 1)
    line = "1 ; Slimer ; creature ; 1 ; 2 ; 1 ; ------ ; 1 ; 0 ; 0 ; 2/1 Creature. Summon: You gain 1 health."  # noqa: E501
    cards = parse_cardlist(line)
    assert len(cards) == 1
    c = cards[0]
    assert c.id == 1
    assert c.name == "Slimer"
    assert c.type == CardType.CREATURE
    assert c.cost == 1
    assert c.attack == 2
    assert c.defense == 1
    assert c.abilities == "------"
    assert c.player_hp == 1
    assert c.enemy_hp == 0
    assert c.card_draw == 0
    assert len(c.abilities) == 6
    assert c.type in set(CardType)


def test_parse_abilities_non_empty():
    # Card 7: Rootkin Sapling has Ward (W)
    line = "7 ; Rootkin Sapling ; creature ; 2 ; 2 ; 2 ; -----W ; 0 ; 0 ; 0 ; 2/2 Creature. Ward."
    cards = parse_cardlist(line)
    assert len(cards) == 1
    c = cards[0]
    assert c.abilities == "-----W"
    assert c.has("W")
    assert not c.has("B")


def test_parse_items():
    # Green item
    line_green = "80 ; Strength Potion ; itemGreen ; 2 ; 2 ; 0 ; ------ ; 0 ; 0 ; 0 ; Green Item. Give a friendly creature +2/+0."  # noqa: E501
    cards_g = parse_cardlist(line_green)
    assert cards_g[0].type == CardType.GREEN_ITEM

    # Red item (negative stats)
    line_red = "110 ; Decimate ; itemRed ; 4 ; 0 ; -6 ; ------ ; 0 ; 0 ; 0 ; Red Item. Give an enemy creature +0/-6."  # noqa: E501
    cards_r = parse_cardlist(line_red)
    assert cards_r[0].type == CardType.RED_ITEM

    # Blue item
    line_blue = "160 ; Minor Life Steal Potion ; itemBlue ; 2 ; 0 ; 0 ; ------ ; 2 ; -2 ; 0 ; Blue Item. Deal 2 damage to your opponent and gain 2 health."  # noqa: E501
    cards_b = parse_cardlist(line_blue)
    assert cards_b[0].type == CardType.BLUE_ITEM


def _row(type_: str, description: str, **kw) -> dict:
    base = {
        "type": type_,
        "description": description,
        "attack": 0,
        "defense": 0,
        "player_hp": 0,
        "enemy_hp": 0,
        "card_draw": 0,
    }
    base.update(kw)
    return base


def test_card_text_creature_special():
    assert card_text(_row("creature", "2/1 Creature. Summon: You gain 1 health.")) == (
        "Summon: You gain 1 health."
    )
    assert card_text(_row("creature", "2/2 Creature.")) == ""  # vanilla
    assert card_text(_row("creature", "2/2 Creature. Ward.")) == ""  # keyword only
    # Blizzard Demon: comma-separated keywords only -> empty
    assert card_text(_row("creature", "2/2 Creature. Charge, Drain.")) == ""
    # Night Howler: comma keywords + a special -> keep only the special
    assert (
        card_text(_row("creature", "6/5 Creature. Breakthrough, Drain. Summon: You lose 3 health."))
        == "Summon: You lose 3 health."
    )


def test_card_text_item_effect():
    # cleaned description (the "<Colour> Item." preface removed)
    assert (
        card_text(_row("itemgreen", "Green Item. Give a friendly creature +1/+1 and Breakthrough."))
        == "Give a friendly creature +1/+1 and Breakthrough."
    )
    # derived stat/HP summary when there is no description text
    assert card_text(_row("itemred", "", defense=-6)) == "0/-6"
    assert card_text(_row("itemblue", "", player_hp=2, enemy_hp=-2)) == "+2♥ · foe -2♥"


def test_catalog_includes_card_text_and_raw_description():
    cards = {c["id"]: c for c in catalog()}
    assert cards[1]["card_text"] == "Summon: You gain 1 health."  # Slimer
    assert cards[3]["card_text"] == ""  # Beavrat (vanilla)
    assert cards[41]["card_text"] == ""  # Blizzard Demon (comma keywords only)
    assert cards[45]["card_text"] == "Summon: You lose 3 health."  # Night Howler
    # raw description is still present (lossless)
    assert cards[1]["description"].startswith("2/1 Creature")
