from locma.core.cards import Card, CardType, ABILITY_ORDER, normalize_abilities

def test_normalize_abilities_from_letters():
    assert normalize_abilities("BG") == "B--G--"
    assert normalize_abilities("------") == "------"
    assert normalize_abilities("BCDGLW") == "BCDGLW"

def test_card_has_ability():
    c = Card(1, "Test", CardType.CREATURE, 2, 3, 2, normalize_abilities("G"), 0, 0, 0)
    assert c.has("G") is True
    assert c.has("L") is False
    assert ABILITY_ORDER == "BCDGLW"
