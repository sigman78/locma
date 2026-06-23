from __future__ import annotations

from locma.core.cards import CardType
from locma.data.cards_db import load_cards, parse_cardlist, card_by_id


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
    line = "1 ; Slimer ; creature ; 1 ; 2 ; 1 ; ------ ; 1 ; 0 ; 0 ; 2/1 Creature. Summon: You gain 1 health."
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
    line_green = "80 ; Strength Potion ; itemGreen ; 2 ; 2 ; 0 ; ------ ; 0 ; 0 ; 0 ; Green Item. Give a friendly creature +2/+0."
    cards_g = parse_cardlist(line_green)
    assert cards_g[0].type == CardType.GREEN_ITEM

    # Red item (negative stats)
    line_red = "110 ; Decimate ; itemRed ; 4 ; 0 ; -6 ; ------ ; 0 ; 0 ; 0 ; Red Item. Give an enemy creature +0/-6."
    cards_r = parse_cardlist(line_red)
    assert cards_r[0].type == CardType.RED_ITEM

    # Blue item
    line_blue = "160 ; Minor Life Steal Potion ; itemBlue ; 2 ; 0 ; 0 ; ------ ; 2 ; -2 ; 0 ; Blue Item. Deal 2 damage to your opponent and gain 2 health."
    cards_b = parse_cardlist(line_blue)
    assert cards_b[0].type == CardType.BLUE_ITEM
