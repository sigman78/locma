from locma.core.cards import Card, CardType, normalize_abilities
from locma.core.instance import CardInstance


def _creature(abilities=""):
    return Card(1, "C", CardType.CREATURE, 2, 3, 2, normalize_abilities(abilities), 0, 0, 0)


def test_from_card_copies_stats():
    inst = CardInstance.from_card(_creature(), instance_id=7)
    assert (inst.attack, inst.defense, inst.instance_id) == (3, 2, 7)
    assert inst.can_attack is False
    assert inst.has_attacked is False


def test_charge_can_attack_immediately():
    inst = CardInstance.from_card(_creature("C"), instance_id=1)
    assert inst.can_attack is True


def test_instance_has_reads_mutable_abilities():
    inst = CardInstance.from_card(_creature("G"), instance_id=1)
    assert inst.has("G") is True
