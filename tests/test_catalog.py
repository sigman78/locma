from __future__ import annotations

from locma.data.cards_db import catalog


def test_catalog_shape():
    cards = catalog()
    assert len(cards) == 160
    c1 = cards[0]
    assert c1["id"] == 1 and c1["name"] == "Slimer" and c1["type"] == "creature"
    assert c1["cost"] == 1 and c1["attack"] == 2 and c1["defense"] == 1
    assert "Summon" in c1["description"]
    assert set(c1) == {
        "id",
        "name",
        "type",
        "cost",
        "attack",
        "defense",
        "abilities",
        "player_hp",
        "enemy_hp",
        "card_draw",
        "description",
    }
