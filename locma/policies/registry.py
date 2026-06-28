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


def _azlite(params, spec):
    """AlphaZero-lite — PUCT-guided MCTS with a heuristic (policy, value) oracle.

    Spec ``azlite:iterations,c_puct,seed,rollout_turns``. Paired with the
    `balanced` draft (the draft sweep's best partner; see docs/baseline.md), as
    `ppo:` is — so the matchup against the baselines is apples-to-apples.
    """
    from locma.policies.azlite import AZLiteBattlePolicy  # noqa: PLC0415

    iters = int(params[0]) if len(params) > 0 else 100
    c_puct = float(params[1]) if len(params) > 1 else 1.5
    seed = int(params[2]) if len(params) > 2 else 0
    rollout_turns = int(params[3]) if len(params) > 3 else 0
    return Composer(
        AZLiteBattlePolicy(iterations=iters, c_puct=c_puct, seed=seed, rollout_turns=rollout_turns),
        BalancedDraftPolicy(),
        name=spec,
    )


def _puct_ppo(params, spec):
    """Experimental diagnostic — PPO-prior perfect-info PUCT.

    Spec ``puct-ppo:iterations,model_path,c_puct,seed,turns,obs``. This is
    hidden from normal policy listings because it is search-at-inference and
    perfect-information unless wrapped by ``dpuct-ppo``.
    """
    from locma.policies.azlite import PUCTPPOBattlePolicy  # noqa: PLC0415

    iters = int(params[0]) if len(params) > 0 else 100
    model_path = params[1] if len(params) > 1 else "model.zip"
    c_puct = float(params[2]) if len(params) > 2 else 1.5
    seed = int(params[3]) if len(params) > 3 else 0
    rollout_turns = int(params[4]) if len(params) > 4 else 0
    obs_mode = params[5] if len(params) > 5 else "base"
    return Composer(
        PUCTPPOBattlePolicy(
            iterations=iters,
            model_path=model_path,
            c_puct=c_puct,
            seed=seed,
            rollout_turns=rollout_turns,
            obs_mode=obs_mode,
        ),
        BalancedDraftPolicy(),
        name=spec,
    )


def _dpuct_ppo(params, spec):
    """Fair determinized PPO-prior PUCT — ``dpuct-ppo:K,I,model,c_puct,seed,turns,obs``."""
    from locma.policies.azlite import DeterminizedPUCTPPOBattlePolicy  # noqa: PLC0415

    k = int(params[0]) if len(params) > 0 else 5
    iters = int(params[1]) if len(params) > 1 else 5
    model_path = params[2] if len(params) > 2 else "model.zip"
    c_puct = float(params[3]) if len(params) > 3 else 1.5
    seed = int(params[4]) if len(params) > 4 else 0
    rollout_turns = int(params[5]) if len(params) > 5 else 0
    obs_mode = params[6] if len(params) > 6 else "base"
    return Composer(
        DeterminizedPUCTPPOBattlePolicy(
            determinizations=k,
            iterations=iters,
            model_path=model_path,
            c_puct=c_puct,
            seed=seed,
            rollout_turns=rollout_turns,
            obs_mode=obs_mode,
        ),
        BalancedDraftPolicy(),
        name=spec,
    )


def _dmcts(params, spec):
    """Determinized (non-cheating) MCTS — spec ``dmcts:K,I,seed,turns,det``."""
    from locma.policies.mcts import DMCTSBattlePolicy  # noqa: PLC0415

    k = int(params[0]) if len(params) > 0 else 15  # determinizations (worlds)
    i = int(params[1]) if len(params) > 1 else 30  # iterations per world
    seed = int(params[2]) if len(params) > 2 else 0
    rollout_turns = int(params[3]) if len(params) > 3 else 3
    deterministic = bool(int(params[4])) if len(params) > 4 else False
    return Composer(
        DMCTSBattlePolicy(
            determinizations=k,
            iterations=i,
            seed=seed,
            rollout_turns=rollout_turns,
            deterministic=deterministic,
        ),
        GreedyDraftPolicy(),
        name=spec,
    )


def _netdmcts(params, spec):
    """Net-guided determinized PUCT — spec ``netdmcts:K,I,c_puct,model_path``.

    Paired with the ``balanced`` draft (same as ``ppo``/``azlite`` for
    apples-to-apples comparisons against the scripted baselines).
    """
    from locma.policies.net_oracle import NetGuidedDMCTSBattlePolicy  # noqa: PLC0415

    k = int(params[0]) if len(params) > 0 else 15
    i = int(params[1]) if len(params) > 1 else 80
    c_puct = float(params[2]) if len(params) > 2 else 1.5
    model_path = params[3] if len(params) > 3 else "model.zip"
    return Composer(
        NetGuidedDMCTSBattlePolicy(
            determinizations=k, iterations=i, c_puct=c_puct, model_path=model_path
        ),
        BalancedDraftPolicy(),
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


def _ppo_tactical(params, spec):
    """Experimental eval-only PPO spec for tactical observation artifacts."""
    from locma.policies.ppo import (  # noqa: PLC0415
        MaskablePPOBattlePolicy,
    )

    model_path = params[0] if params else "model.zip"
    return Composer(
        MaskablePPOBattlePolicy(model_path=model_path, obs_mode="tactical"),
        BalancedDraftPolicy(),
        name=spec,
    )


# The pool of baseline opponents a `mixed` training opponent draws from.
_MIXED_POOL = ("random", "scripted", "greedy", "max-guard", "max-attack")


def _mixed(params, spec):
    from locma.policies.mixed import MixedOpponentPolicy  # noqa: PLC0415

    seed = int(params[0]) if params else 0
    pool = [make_policy(b) for b in _MIXED_POOL]
    return MixedOpponentPolicy(pool, name=spec, seed=seed)


_RICH_MIXED_POOL: tuple[tuple[str, float], ...] = (
    ("scripted", 2.0),
    ("greedy", 2.0),
    ("max-guard", 2.0),
    ("max-attack", 2.0),
    ("dmcts:2,3,0,3", 1.0),
    ("dmcts:4,6,1,3", 1.0),
)


def _weighted_mixed(params, spec, pool_specs):
    from locma.policies.mixed import MixedOpponentPolicy  # noqa: PLC0415

    seed = int(params[0]) if params else 0
    specs, weights = zip(*pool_specs, strict=True)
    pool = [make_policy(s) for s in specs]
    return MixedOpponentPolicy(pool, name=spec, seed=seed, weights=weights)


def _mixed_rich(params, spec):
    return _weighted_mixed(params, spec, _RICH_MIXED_POOL)


# Registration order matters: drives policy_names() and table/tournament order.
_FACTORIES = {
    "random": _random,
    "scripted": _scripted,
    "greedy": _greedy,
    "max-guard": _max_guard,
    "max-attack": _max_attack,
    "mcts": _mcts,
    "azlite": _azlite,
    "puct-ppo": _puct_ppo,
    "dpuct-ppo": _dpuct_ppo,
    "dmcts": _dmcts,
    "netdmcts": _netdmcts,
    "ppo": _ppo,
    "ppo-tactical": _ppo_tactical,
    "mixed": _mixed,
    "mixed-rich": _mixed_rich,
}

# Not offered as bare selectable names (e.g. in the server dropdown):
# `ppo`/`puct-ppo`/`netdmcts` need model artifacts + the [ml] extra (use explicit
# specs); `mixed` variants are non-stationary training opponents.
_HIDDEN = {
    "ppo",
    "ppo-tactical",
    "netdmcts",
    "puct-ppo",
    "dpuct-ppo",
    "mixed",
    "mixed-rich",
}


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
