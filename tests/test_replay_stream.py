from __future__ import annotations

from locma.policies.registry import make_policy


def test_make_policy_known():
    p = make_policy("greedy")
    assert p.name == "greedy"


def test_make_policy_unknown():
    import pytest
    with pytest.raises(ValueError):
        make_policy("nope")
