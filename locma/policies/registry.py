from __future__ import annotations

from locma.policies.greedy import GreedyPolicy
from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy

_REGISTRY = {"random": RandomPolicy, "scripted": ScriptedPolicy, "greedy": GreedyPolicy}


def make_policy(spec: str):
    """Construct a built-in policy by name. Raises ValueError on unknown spec."""
    if spec in _REGISTRY:
        return _REGISTRY[spec](spec)
    raise ValueError(f"unknown policy '{spec}'")
