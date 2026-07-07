from __future__ import annotations

import math

from locma.depot import resolve_path
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
from locma.policies.exploits import ShellBattlePolicy, ShellDraftPolicy


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


def _dmcts(params, spec):
    """Determinized (non-cheating) MCTS — spec ``dmcts:K,I,seed,turns``."""
    from locma.policies.mcts import DMCTSBattlePolicy  # noqa: PLC0415

    k = int(params[0]) if len(params) > 0 else 15  # determinizations (worlds)
    i = int(params[1]) if len(params) > 1 else 30  # iterations per world
    seed = int(params[2]) if len(params) > 2 else 0
    rollout_turns = int(params[3]) if len(params) > 3 else 3
    return Composer(
        DMCTSBattlePolicy(determinizations=k, iterations=i, seed=seed, rollout_turns=rollout_turns),
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
    model_path = resolve_path(params[3] if len(params) > 3 else "model.zip")
    return Composer(
        NetGuidedDMCTSBattlePolicy(
            determinizations=k, iterations=i, c_puct=c_puct, model_path=model_path
        ),
        BalancedDraftPolicy(),
        name=spec,
    )


def _vbeam(params, spec):
    """V-greedy own-turn beam planner — spec ``vbeam:model_path,width,max_actions,item_discount``.

    Plans whole turns by beam-searching own-turn action sequences and scoring
    stopping points with the token model's value head (E5 "planning-lite").
    Model path first so the common case (``vbeam:depot:b0/b0_s0.zip``) needs
    no commas. ``model_path`` may also be ``|``-separated paths
    (``vbeam:a.zip|b.zip|c.zip``) — the beam then ranks with the mean of the
    member critics (``EnsembleValueEvaluator``). Paired with the ``balanced``
    draft like ``ppo``/``azlite``/``netdmcts`` for apples-to-apples
    comparisons. ``item_discount`` overrides the balanced draft's item
    discount (default 12, tuned for the REACTIVE pilot; the planner converts
    items far better — E16a/E17).
    """
    from locma.policies.vbeam import EnsembleValueEvaluator, VBeamBattlePolicy  # noqa: PLC0415

    raw = params[0] if len(params) > 0 and params[0] else "model.zip"
    width = int(params[1]) if len(params) > 1 else 8
    max_actions = int(params[2]) if len(params) > 2 else 20
    item_discount = float(params[3]) if len(params) > 3 else None
    if "|" in raw:
        paths = [resolve_path(p) for p in raw.split("|")]
        battle = VBeamBattlePolicy(
            model_path=paths[0],
            width=width,
            max_actions=max_actions,
            evaluator=EnsembleValueEvaluator(paths),
        )
    else:
        battle = VBeamBattlePolicy(
            model_path=resolve_path(raw), width=width, max_actions=max_actions
        )
    draft = BalancedDraftPolicy() if item_discount is None else BalancedDraftPolicy(item_discount=item_discount)
    return Composer(battle, draft, name=spec)


def _ppo(params, spec):
    from locma.policies.ppo import (  # noqa: PLC0415
        MaskablePPOBattlePolicy,
    )

    model_path = resolve_path(params[0] if params else "model.zip")
    # Pair the learned battle net with a `balanced` draft, not `greedy`: the draft
    # sweep (docs/baseline.md "PPO × draft sweep") found the greedy draft is the
    # WORST partner (0.39 avg vs the ground baselines) while `balanced` (0.54) makes
    # the same battle net BEAT them. The battle policy is deck-robust, so this needs
    # no retraining. Optional second param overrides the balanced item discount
    # (``ppo:path,3``) — E17 guard-rail arms.
    item_discount = float(params[1]) if len(params) > 1 else None
    draft = BalancedDraftPolicy() if item_discount is None else BalancedDraftPolicy(item_discount=item_discount)
    return Composer(MaskablePPOBattlePolicy(model_path=model_path), draft, name=spec)


# --- E10 exploit archetypes: scripted strategies aimed at the learned
# policies' suspected blind spots (see locma/policies/exploits.py). ---


def _rnddeck(params, spec):
    """Exploit probe (a): a competent aggressive pilot on a RANDOM deck —
    out-of-distribution decks for policies tuned on curated drafts."""
    seed = int(params[0]) if params else 0
    return Composer(GroundBattlePolicy(), RandomDraftPolicy(seed=seed), name=spec)


def _guardwall(params, spec):
    """Exploit probe (b): Guard wall shielding high-attack threats; face only."""
    return Composer(
        ShellBattlePolicy(use_green=False, value_trades=False),
        ShellDraftPolicy(use_green=False),
        name=spec,
    )


def _bufface(params, spec):
    """Exploit probe (c): green buffs on the biggest attacker, then face."""
    return Composer(
        ShellBattlePolicy(use_green=True, value_trades=False),
        ShellDraftPolicy(use_green=True),
        name=spec,
    )


def _boardkeep(params, spec):
    """Exploit probe (d): board preservation — winning trades only, balanced deck."""
    return Composer(
        ShellBattlePolicy(use_green=False, value_trades=True),
        BalancedDraftPolicy(),
        name=spec,
    )


def _shell(params, spec):
    """The full exploit package: wall + buffs + winning trades + removal."""
    return Composer(
        ShellBattlePolicy(use_green=True, value_trades=True),
        ShellDraftPolicy(use_green=True),
        name=spec,
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
    "azlite": _azlite,
    "dmcts": _dmcts,
    "netdmcts": _netdmcts,
    "vbeam": _vbeam,
    "ppo": _ppo,
    "mixed": _mixed,
    "rnddeck": _rnddeck,
    "guardwall": _guardwall,
    "bufface": _bufface,
    "boardkeep": _boardkeep,
    "shell": _shell,
}

# Not offered as bare selectable names (e.g. in the server dropdown):
# `ppo`, `netdmcts` and `vbeam` need a model artifact + the [ml] extra (use
# `ppo:path`, `netdmcts:K,I,c,path` or `vbeam:path,width,max_actions`);
# `mixed` is a non-stationary training opponent, not a baseline to rank.
_HIDDEN = {"ppo", "mixed", "netdmcts", "vbeam"}


def policy_names() -> list[str]:
    """Selectable built-in policy names, in registration order."""
    return [n for n in _FACTORIES if n not in _HIDDEN]


def is_policy_spec(s: str) -> bool:
    """True when ``s`` names a registered policy (``base`` or ``base:params``).

    A bare model path is NOT a spec: ``runs/b0_s0.zip`` has no known base, and
    a Windows drive prefix (``F:\\...``) parses to a base like ``F`` which is
    not registered either.
    """
    return s.partition(":")[0] in _FACTORIES


def make_policy(spec: str):
    """Construct a built-in policy from a spec string ``base[:p1,p2,...]``.

    Parameters are positional and split on the first colon only (so paths with
    colons survive); trailing parameters fall back to per-preset defaults.
    Model-path parameters accept plain files or ``depot:`` refs (resolved via
    ``locma.depot.resolve_path``), e.g. ``vbeam:depot:b0/b0_s0.zip``.
    Raises ValueError on an unknown base name.
    """
    base, sep, paramstr = spec.partition(":")
    params = paramstr.split(",") if sep and paramstr else []
    if base in _FACTORIES:
        return _FACTORIES[base](params, spec)
    raise ValueError(f"unknown policy '{spec}'")
