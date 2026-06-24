from __future__ import annotations

from locma.policies.baselines import MaxAttackDraftPolicy, MaxGuardDraftPolicy
from locma.policies.greedy import GreedyPolicy
from locma.policies.random_policy import RandomPolicy
from locma.policies.scripted import ScriptedPolicy

_REGISTRY = {
    "random": RandomPolicy,
    "scripted": ScriptedPolicy,
    "greedy": GreedyPolicy,
    "max-guard": MaxGuardDraftPolicy,
    "max-attack": MaxAttackDraftPolicy,
}


def policy_names() -> list[str]:
    """Names of all built-in policies, in registration order."""
    return list(_REGISTRY)


def make_policy(spec: str):
    """Construct a built-in policy by name. Raises ValueError on unknown spec."""
    if spec in _REGISTRY:
        return _REGISTRY[spec](spec)
    raise ValueError(f"unknown policy '{spec}'")
