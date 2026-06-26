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
    BalancedDraftPolicy,
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
    rollout_turns = int(params[3]) if len(params) > 3 else 3  # heuristic rollout (0 = legacy)
    return Composer(
        MCTSBattlePolicy(iterations=iters, c=c, seed=seed, rollout_turns=rollout_turns),
        GreedyDraftPolicy(),
        name=spec,
    )


def _ppo(params, spec):
    from locma.policies.ppo import (  # noqa: PLC0415
        MaskablePPOBattlePolicy,
    )

    model_path = params[0] if params else "model.zip"
    # Pair the learned battle net with a `balanced` draft, not `greedy`: the draft
    # sweep (docs/baseline.md "PPO × draft sweep") found the greedy draft is the
    # WORST partner (0.39 avg vs the ground baselines) while `balanced` (0.54) makes
    # the same battle net BEAT them. The battle policy is deck-robust, so this needs
    # no retraining.
    return Composer(
        MaskablePPOBattlePolicy(model_path=model_path), BalancedDraftPolicy(), name=spec
    )


# The pool of baseline opponents a `mixed` training opponent draws from.
_MIXED_POOL = ("random", "scripted", "greedy", "max-guard", "max-attack")


def _mixed(params, spec):
    from locma.policies.mixed import MixedOpponentPolicy  # noqa: PLC0415

    seed = int(params[0]) if params else 0
    pool = [make_policy(b) for b in _MIXED_POOL]
    return MixedOpponentPolicy(pool, name=spec, seed=seed)


# Registration order matters: drives policy_names() and table/tournament order.
_FACTORIES = {
    "random": _random,
    "scripted": _scripted,
    "greedy": _greedy,
    "max-guard": _max_guard,
    "max-attack": _max_attack,
    "mcts": _mcts,
    "ppo": _ppo,
    "mixed": _mixed,
}

# Not offered as bare selectable names (e.g. in the server dropdown):
# `ppo` needs a model artifact + the [ml] extra (use `ppo:path`); `mixed` is a
# non-stationary training opponent, not a baseline to rank.
_HIDDEN = {"ppo", "mixed"}


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
