from __future__ import annotations

import math
import os

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
    DistilledDraftPolicy,
    GreedyDraftPolicy,
    MaxAttackDraftPolicy,
    MaxDefenseDraftPolicy,
    MaxGuardDraftPolicy,
    RandomDraftPolicy,
    WeightedDraftPolicy,
)
from locma.policies.exploits import ShellBattlePolicy, ShellDraftPolicy


def _draft_param(params, i):
    """Resolve the optional draft-override parameter of ``ppo:``/``vbeam:`` specs.

    Absent/empty -> the default balanced draft. A float -> balanced with that
    item discount (E17, e.g. ``ppo:model.zip,3``). A ``.json`` path with a
    "values" key -> a per-card priority table, loaded as DistilledDraftPolicy
    -- either fit by regression against a net's revealed picks, or elicited
    directly from the net via neutral-context comparisons and spliced onto a
    curve_target (E20; the JSON may carry its own curve_target/w_need/
    w_creature, else BalancedDraftPolicy's defaults apply). A ``.json`` path
    WITHOUT "values" -> a hand-specified curve/discount override (E20
    census-derived heuristic, ``{"curve_target": ..., "item_discount":
    ...}``, loaded as BalancedDraftPolicy with those constants). Anything else
    -> a learned draft model path (E18b, e.g. ``ppo:model.zip,runs/draft_s0.
    zip`` or a ``depot:`` ref), loaded lazily as MaskablePPODraftPolicy.
    """
    if len(params) <= i or not params[i]:
        return BalancedDraftPolicy()
    try:
        return BalancedDraftPolicy(item_discount=float(params[i]))
    except ValueError:
        path = resolve_path(params[i])
        if path.endswith(".json"):
            import json  # noqa: PLC0415

            with open(path, encoding="utf-8") as f:
                spec = json.load(f)
            if "values" in spec:
                return DistilledDraftPolicy.load(path)
            return BalancedDraftPolicy(
                item_discount=spec["item_discount"],
                curve_target={int(k): v for k, v in spec["curve_target"].items()},
            )

        from locma.policies.ppo import MaskablePPODraftPolicy  # noqa: PLC0415

        return MaskablePPODraftPolicy(model_path=path)


def _greedy_draft_param(params, i):
    """Like ``_draft_param``, but defaults to ``GreedyDraftPolicy`` (mcts/dmcts's
    historic, hardcoded default) instead of balanced when the param is absent —
    so old bare ``mcts:100``/``dmcts:15,30`` specs stay byte-identical, while a
    5th param opts into a draft override (E22: same-draft head-to-head vs vbeam)."""
    if len(params) <= i or not params[i]:
        return GreedyDraftPolicy()
    return _draft_param(params, i)


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
    """Cheating perfect-information MCTS — spec ``mcts:iterations,c,seed,turns,draft``.

    Defaults to the ``greedy`` draft (historic default, kept for reproducibility
    of old ``mcts:100``-style specs). The optional 5th param overrides the draft
    the same way ``ppo:``/``vbeam:`` do (see ``_draft_param``) — e.g.
    ``mcts:1000,,,,depot:ldraft/ldraft_s0.zip`` for a same-draft comparison
    against a learned-draft opponent (E22).
    """
    from locma.policies.mcts import (  # noqa: PLC0415
        MCTSBattlePolicy,
    )

    iters = int(params[0]) if len(params) > 0 else 100
    c = float(params[1]) if len(params) > 1 else math.sqrt(2)
    seed = int(params[2]) if len(params) > 2 else 0
    rollout_turns = int(params[3]) if len(params) > 3 else 3  # heuristic rollout (0 = legacy)
    return Composer(
        MCTSBattlePolicy(iterations=iters, c=c, seed=seed, rollout_turns=rollout_turns),
        _greedy_draft_param(params, 4),
        name=spec,
    )


def _azlite(params, spec):
    """AlphaZero-lite — PUCT-guided MCTS with a heuristic (policy, value) oracle.

    Spec ``azlite:iterations,c_puct,seed,rollout_turns,draft``. Defaults to the
    `balanced` draft (the draft sweep's best partner; see docs/baseline.md), as
    `ppo:` is — so the matchup against the baselines is apples-to-apples. The
    optional 5th param overrides the draft like ``ppo``/``vbeam``/``netdmcts``
    (see ``_draft_param``) so search can be compared on a matched deck
    distribution (E25 strong-league, same-draft head-to-head). NB: azlite
    *cheats* (perfect foresight over future draws) — a matched-draft comparison
    isolates that information edge, it does not make it fair.
    """
    from locma.policies.azlite import AZLiteBattlePolicy  # noqa: PLC0415

    iters = int(params[0]) if len(params) > 0 else 100
    c_puct = float(params[1]) if len(params) > 1 else 1.5
    seed = int(params[2]) if len(params) > 2 else 0
    rollout_turns = int(params[3]) if len(params) > 3 else 0
    return Composer(
        AZLiteBattlePolicy(iterations=iters, c_puct=c_puct, seed=seed, rollout_turns=rollout_turns),
        _draft_param(params, 4),
        name=spec,
    )


def _dmcts(params, spec):
    """Determinized (non-cheating) MCTS — spec ``dmcts:K,I,seed,turns,draft``.

    Defaults to the ``greedy`` draft (historic default). The optional 5th
    param overrides the draft like ``mcts:`` (see ``_greedy_draft_param``)."""
    from locma.policies.mcts import DMCTSBattlePolicy  # noqa: PLC0415

    k = int(params[0]) if len(params) > 0 else 15  # determinizations (worlds)
    i = int(params[1]) if len(params) > 1 else 30  # iterations per world
    seed = int(params[2]) if len(params) > 2 else 0
    rollout_turns = int(params[3]) if len(params) > 3 else 3
    return Composer(
        DMCTSBattlePolicy(determinizations=k, iterations=i, seed=seed, rollout_turns=rollout_turns),
        _greedy_draft_param(params, 4),
        name=spec,
    )


def _netdmcts(params, spec):
    """Net-guided determinized PUCT — spec ``netdmcts:K,I,c_puct,model_path,draft``.

    Defaults to the ``balanced`` draft (same as ``ppo``/``azlite`` for
    reproducibility of historical specs). The optional 5th param overrides
    the draft like ``ppo``/``vbeam`` so search depth can be compared on the
    same deck distribution (E23).
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
        _draft_param(params, 4),
        name=spec,
    )


def _vbeam(params, spec):
    """V-greedy own-turn beam planner — spec ``vbeam:model_path,width,max_actions,draft``.

    Plans whole turns by beam-searching own-turn action sequences and scoring
    stopping points with the token model's value head (E5 "planning-lite").
    Model path first so the common case (``vbeam:depot:b0/b0_s0.zip``) needs
    no commas. ``model_path`` may also be ``|``-separated paths
    (``vbeam:a.zip|b.zip|c.zip``) — the beam then ranks with the mean of the
    member critics (``EnsembleValueEvaluator``). Paired with the ``balanced``
    draft like ``ppo``/``azlite``/``netdmcts`` for apples-to-apples
    comparisons. The 4th param overrides the draft: a float sets the balanced
    item discount (default 12, tuned for the REACTIVE pilot; the planner
    converts items far better — E16a/E17), a model path loads a learned draft
    (E18b). See ``_draft_param``.
    """
    from locma.policies.vbeam import EnsembleValueEvaluator, VBeamBattlePolicy  # noqa: PLC0415

    raw = params[0] if len(params) > 0 and params[0] else "model.zip"
    width = int(params[1]) if len(params) > 1 else 8
    max_actions = int(params[2]) if len(params) > 2 else 20
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
    return Composer(battle, _draft_param(params, 3), name=spec)


def _rbeam(params, spec):
    """Reply-aware turn beam — spec ``rbeam:model,width,max_actions,n_plans,n_worlds,draft``.

    ``vbeam`` plus one genuine opponent-reply ply: turn-level expectiminimax over
    the beam's top ``n_plans`` own-turn plans, averaged across ``n_worlds`` fair
    determinizations (E22/E23 — depth beats hidden-info; see locma/policies/
    rbeam.py). ``model`` may be ``|``-separated for the shared-critic ensemble,
    exactly like ``vbeam``; the same ensemble also models the opponent's reply.
    The 6th param overrides the draft (``_draft_param``) so search can be scored
    at a fixed deck distribution against the planner.
    """
    from locma.policies.rbeam import RBeamBattlePolicy  # noqa: PLC0415
    from locma.policies.vbeam import EnsembleValueEvaluator  # noqa: PLC0415

    raw = params[0] if len(params) > 0 and params[0] else "model.zip"
    width = int(params[1]) if len(params) > 1 and params[1] else 8
    max_actions = int(params[2]) if len(params) > 2 and params[2] else 20
    n_plans = int(params[3]) if len(params) > 3 and params[3] else 4
    n_worlds = int(params[4]) if len(params) > 4 and params[4] else 4
    kw = {"width": width, "max_actions": max_actions, "n_plans": n_plans, "n_worlds": n_worlds}
    if "|" in raw:
        paths = [resolve_path(p) for p in raw.split("|")]
        battle = RBeamBattlePolicy(
            model_path=paths[0], evaluator=EnsembleValueEvaluator(paths), **kw
        )
    else:
        battle = RBeamBattlePolicy(model_path=resolve_path(raw), **kw)
    return Composer(battle, _draft_param(params, 5), name=spec)


def _ppo(params, spec):
    from locma.policies.ppo import (  # noqa: PLC0415
        MaskablePPOBattlePolicy,
        MaskablePPOEnsembleBattlePolicy,
    )

    # Pair the learned battle net with a `balanced` draft, not `greedy`: the draft
    # sweep (docs/baseline.md "PPO × draft sweep") found the greedy draft is the
    # WORST partner (0.39 avg vs the ground baselines) while `balanced` (0.54) makes
    # the same battle net BEAT them. The battle policy is deck-robust, so this needs
    # no retraining. Optional second param overrides the draft: a float sets the
    # balanced item discount (``ppo:path,3`` — E17 guard-rail arms), a model path
    # loads a learned draft (``ppo:path,draft.zip`` — E18b). See ``_draft_param``.
    # `model` may also be `|`-separated paths (``ppo:a.zip|b.zip|c.zip``), same
    # idiom as ``vbeam:`` — the battle half is then the mean-of-policy-heads
    # ensemble (E26, ``MaskablePPOEnsembleBattlePolicy``) instead of one net.
    raw = params[0] if params else "model.zip"
    if "|" in raw:
        paths = [resolve_path(p) for p in raw.split("|")]
        battle = MaskablePPOEnsembleBattlePolicy(paths)
    else:
        battle = MaskablePPOBattlePolicy(model_path=resolve_path(raw))
    return Composer(battle, _draft_param(params, 1), name=spec)


def _lppo(params, spec):
    """Lethal-guarded PPO — spec ``lppo:model[,draft[,node_cap]]`` (E26/E14a).

    ``ppo:`` (same ``model``/``draft`` parsing, including the ``|``-separated
    ensemble form) wrapped in ``LethalGuardBattlePolicy``: an exhaustive,
    zero-training own-turn lethal solver that plays a forced win when one
    exists this turn and otherwise gets out of the way (see
    ``locma.policies.lguard`` for the fairness/soundness argument). ``model``
    may be a single path or ``|``-separated paths (ensemble inner). The
    optional 3rd param overrides the solver's DFS node cap (default 3000).
    """
    from locma.policies.lguard import LethalGuardBattlePolicy  # noqa: PLC0415
    from locma.policies.ppo import (  # noqa: PLC0415
        MaskablePPOBattlePolicy,
        MaskablePPOEnsembleBattlePolicy,
    )

    raw = params[0] if params and params[0] else "model.zip"
    node_cap = int(params[2]) if len(params) > 2 and params[2] else 3000
    if "|" in raw:
        paths = [resolve_path(p) for p in raw.split("|")]
        inner_battle = MaskablePPOEnsembleBattlePolicy(paths)
    else:
        inner_battle = MaskablePPOBattlePolicy(model_path=resolve_path(raw))
    battle = LethalGuardBattlePolicy(inner_battle, node_cap=node_cap)
    return Composer(battle, _draft_param(params, 1), name=spec)


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
    "rbeam": _rbeam,
    "ppo": _ppo,
    "lppo": _lppo,
    "mixed": _mixed,
    "rnddeck": _rnddeck,
    "guardwall": _guardwall,
    "bufface": _bufface,
    "boardkeep": _boardkeep,
    "shell": _shell,
}

# Not offered as bare selectable names (e.g. in the server dropdown):
# `ppo`, `lppo`, `netdmcts`, `vbeam` and `rbeam` need a model artifact + the
# [ml] extra (use `ppo:path`, `lppo:path`, `netdmcts:K,I,c,path`,
# `vbeam:path,width,max_actions` or `rbeam:path,width,max_actions,n_plans,
# n_worlds`); `mixed` is a non-stationary training opponent, not a baseline
# to rank.
_HIDDEN = {"ppo", "lppo", "mixed", "netdmcts", "vbeam", "rbeam"}


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


# -- standalone draft policies (the web Play draft's "complete for me") -------

# name -> (dropdown label, factory). Order drives the dropdown; `balanced` first
# because it is the strongest heuristic draft (Condorcet under strong fair
# pilots, docs/baseline.md) and the default.
_DRAFT_FACTORIES = {
    "balanced": ("Balanced — curve-aware (best heuristic)", BalancedDraftPolicy),
    "weighted": ("Weighted — keyword-valued greedy", WeightedDraftPolicy),
    "greedy": ("Greedy — raw stats", GreedyDraftPolicy),
    "max-guard": ("Max Guard — defensive wall", MaxGuardDraftPolicy),
    "max-attack": ("Max Attack — aggro stats", MaxAttackDraftPolicy),
    "max-defense": ("Max Defense — tanky board", MaxDefenseDraftPolicy),
    "random": ("Random", RandomDraftPolicy),
}


def draft_policy_choices() -> list[dict]:
    """``[{name, label}]`` for UI dropdowns — heuristic drafts only; the server
    appends any locally available depot draft nets (it knows the depot)."""
    return [{"name": n, "label": label} for n, (label, _) in _DRAFT_FACTORIES.items()]


def make_draft_policy(spec: str):
    """A standalone draft policy from a dropdown name or a draft-override tail.

    Known names resolve via ``_DRAFT_FACTORIES``; anything else goes through the
    ``_draft_param`` grammar (float item-discount, ``.json`` table/curve, or a
    learned model path / ``depot:`` ref -> MaskablePPODraftPolicy). Unlike
    ``_draft_param``, a model path is validated eagerly (the policy itself is
    lazy) so a bad spec fails the request instead of the mid-draft pick.
    Raises ValueError on an unknown name / missing file.
    """
    entry = _DRAFT_FACTORIES.get(spec)
    if entry is not None:
        return entry[1]()
    try:
        float(spec)  # a numeric item-discount tail needs no file
    except ValueError:
        try:
            path = resolve_path(spec)
        except Exception as e:
            raise ValueError(f"unknown draft policy {spec!r}: {e}") from e
        if not os.path.exists(path):
            raise ValueError(f"unknown draft policy or missing file: {spec!r}") from None
    return _draft_param([spec], 0)
