from locma.core.actions import Attack, Pass, Summon, Use, action_from_dict, action_to_dict


def test_action_roundtrip():
    for action in (Summon(5), Attack(3, -1), Attack(3, 7), Use(2, -1), Use(2, 9), Pass()):
        assert action_from_dict(action_to_dict(action)) == action


def test_action_to_dict_shapes():
    assert action_to_dict(Summon(5)) == {"t": "summon", "id": 5}
    assert action_to_dict(Attack(3, -1)) == {"t": "attack", "a": 3, "target": -1}
    assert action_to_dict(Use(2, 9)) == {"t": "use", "item": 2, "target": 9}
    assert action_to_dict(Pass()) == {"t": "pass"}
