from __future__ import annotations

import pytest

from locma.policies.registry import make_policy


def test_make_policy_known():
    for spec in ("random", "scripted", "greedy"):
        assert make_policy(spec).name == spec


def test_make_policy_unknown():
    with pytest.raises(ValueError, match=r"unknown policy 'nope'"):
        make_policy("nope")
