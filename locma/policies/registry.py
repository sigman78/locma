from __future__ import annotations

import math

from locma.policies.battles import (
    GreedyBattlePolicy,
    GroundBattlePolicy,
    RandomBattlePolicy,
    ScriptedBattlePolicy,
)
from locma.policies.composer import Composer
from locma.policies.drafts import (
    GreedyDraftPolicy,
    MaxAttackDraftPolicy,
    MaxGuardDraftPolicy,
    RandomDraftPolicy,
)


def _random(params, spec):
    return Composer(RandomBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name=spec)


def _scripted(params, spec):
    return Composer(ScriptedBattlePolicy(seed=0), RandomDraftPolicy(seed=0), name=spec)


def _greedy(params, spec):
    return Composer(GreedyBattlePolicy(), GreedyDraftPolicy(), name=spec)


def _max_guard(params, spec):
    return Composer(GroundBattlePolicy(), MaxGuardDraftPolicy(), name=spec)


def _max_attack(params, spec):
    return Composer(GroundBattlePolicy(), MaxAttackDraftPolicy(), name=spec)


def _mcts(params, spec):
    from locma.policies.mcts import (  # noqa: PLC0415
        MCTSBattlePolicy,
    )

    iters = int(params[0]) if len(params) > 0 else 100
    c = float(params[1]) if len(params) > 1 else math.sqrt(2)
    seed = int(params[2]) if len(params) > 2 else 0
    return Composer(
        MCTSBattlePolicy(iterations=iters, c=c, seed=seed), GreedyDraftPolicy(), name=spec
    )


def _ppo(params, spec):
    from locma.policies.ppo import (  # noqa: PLC0415
        MaskablePPOBattlePolicy,
    )

    model_path = params[0] if params else "model.zip"
    return Composer(MaskablePPOBattlePolicy(model_path=model_path), GreedyDraftPolicy(), name=spec)


# Registration order matters: drives policy_names() and table/tournament order.
_FACTORIES = {
    "random": _random,
    "scripted": _scripted,
    "greedy": _greedy,
    "max-guard": _max_guard,
    "max-attack": _max_attack,
    "mcts": _mcts,
    "ppo": _ppo,
}

# ppo needs a model artifact + the [ml] extra, so it is not offered as a bare
# selectable name (e.g. in the server dropdown); use `ppo:path` explicitly.
_HIDDEN = {"ppo"}


def policy_names() -> list[str]:
    """Selectable built-in policy names, in registration order."""
    return [n for n in _FACTORIES if n not in _HIDDEN]


def make_policy(spec: str):
    """Construct a built-in policy from a spec string ``base[:p1,p2,...]``.

    Parameters are positional and split on the first colon only (so paths with
    colons survive); trailing parameters fall back to per-preset defaults.
    Raises ValueError on an unknown base name.
    """
    base, sep, paramstr = spec.partition(":")
    params = paramstr.split(",") if sep and paramstr else []
    if base in _FACTORIES:
        return _FACTORIES[base](params, spec)
    raise ValueError(f"unknown policy '{spec}'")
