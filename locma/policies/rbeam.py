"""Reply-aware turn beam planner ("rbeam", Priority 2 in plans-2026-07-09).

``vbeam`` searches only the current player's own turn and never crosses
``Pass``. E22/E23 showed that multi-turn DEPTH — not hidden-information access —
is the lever that beats the one-turn planner: fair determinized search overtook
the 0.978 ``vbeam`` recipe after only ~1-2 turns of genuine tree depth. ``rbeam``
adds exactly that one missing ply: turn-level expectiminimax over the beam's
best own-turn plans.

For each of the top-N own-turn plans (from ``vbeam``'s beam), sample K plausible
hidden worlds (fair determinization — same machinery as ``dmcts``/``netdmcts``),
end the turn, let the opponent play its strongest complete reply (its own beam
in that world), and score the resulting back-to-us state with the critic. Pick
the root plan with the best mean value across worlds.

Fair by construction: the root plan is FIXED across all K worlds (we commit to
one line, then average over what the opponent might hold), while the opponent
adapts to the hand it actually knows in each sampled world. Reuses the strongest
confirmed assets — the ``shared`` critic ensemble and ``ldraft`` — with no
retraining. Cost sits in the gap between ``vbeam`` (~0.7 s/game) and ``netdmcts``
(tens of s/game): one initial beam plus N*K opponent-reply beams per turn.

Import-safe without the [ml] extra when an evaluator is injected (only the
``EnsembleValueEvaluator`` default touches the ML stack, lazily).
"""

from __future__ import annotations

import random

import numpy as np

from locma.core import battle as battlemod
from locma.core.actions import Pass
from locma.core.engine import make_battle_view
from locma.core.state import Phase
from locma.data.cards_db import load_cards
from locma.policies.mcts import determinize
from locma.policies.vbeam import (
    _LOSS_SCORE,
    _WIN_SCORE,
    NetValueEvaluator,
    plan_turn,
    plan_turn_candidates,
)


def _apply_plan(state, plan, seat) -> None:
    """Apply an own-turn action sequence to ``state`` in place, stopping at game end.

    Actions reference card instances by integer id (see ``core.actions``), and
    ``determinize`` preserves own-side ids + the opponent's visible board, so a
    plan computed on the real root replays cleanly on a determinized world. The
    legality guard is belt-and-suspenders: own-turn moves never depend on the
    opponent's hidden hand or on our deck order, so they stay legal by
    construction — if one somehow isn't, we stop the line there.
    """
    for a in plan:
        if state.phase == Phase.ENDED:
            return
        if a not in battlemod.battle_legal(state):
            break
        battlemod.apply_battle(state, a)


def _simulate_to_leaf(det, plan, seat, opp_evaluator, *, width, max_actions):
    """Play our ``plan`` then the opponent's best reply in ``det`` (mutated).

    Returns ``(terminal_score, view, sign)``:

    - a line that ends the game -> ``(WIN/LOSS sentinel, None, 0)``;
    - a non-terminal line -> ``(None, back_to_us_view, +1|-1)``, where the caller
      scores the view with the critic and multiplies by ``sign``. ``make_battle_view``
      yields the current player's view and the critic value is from that owner's
      perspective; ``sign`` is ``-1`` in the (defensive, should-not-happen) case
      that control is not back on our seat, else ``+1``.

    Separating "reach the leaf" from "score the leaf" lets the caller batch every
    non-terminal leaf across all plans/worlds into ONE critic forward.
    """
    _apply_plan(det, plan, seat)
    if det.phase == Phase.ENDED:
        return (_WIN_SCORE if det.winner == seat else _LOSS_SCORE), None, 0
    # If the plan broke before its terminal Pass, end our turn explicitly so the
    # opponent gets to reply (Pass triggers their hidden draw — now world-fair).
    if det.current == seat:
        battlemod.apply_battle(det, Pass())
        if det.phase == Phase.ENDED:
            return (_WIN_SCORE if det.winner == seat else _LOSS_SCORE), None, 0

    # Opponent's strongest complete reply in this world (its own own-turn beam).
    opp_plan = plan_turn(det, opp_evaluator, width=width, max_actions=max_actions)
    _apply_plan(det, opp_plan, 1 - seat)
    if det.phase == Phase.ENDED:
        return (_WIN_SCORE if det.winner == seat else _LOSS_SCORE), None, 0

    return None, make_battle_view(det), (1 if det.current == seat else -1)


def score_plan_in_world(det, plan, seat, evaluator, opp_evaluator, *, width, max_actions) -> float:
    """One expectiminimax leaf: our ``plan`` -> opponent's best reply -> critic.

    ``det`` is a determinized world (fresh clone) that this function mutates.
    Returns a value in the critic's [-1, 1] range from ``seat``'s perspective, or
    the out-of-range win/loss sentinels when the line ends the game. The batched
    ``plan_turn_reply_aware`` scores its leaves directly; this wrapper is the
    single-leaf equivalent (used by tests and callers that want one world).
    """
    score, view, sign = _simulate_to_leaf(
        det, plan, seat, opp_evaluator, width=width, max_actions=max_actions
    )
    if view is None:
        return score
    return sign * float(evaluator.values([view])[0])


def plan_turn_reply_aware(
    state,
    evaluator,
    *,
    cards,
    rng,
    width: int = 8,
    max_actions: int = 20,
    n_plans: int = 4,
    n_worlds: int = 4,
    opp_evaluator=None,
) -> list:
    """Pick the own-turn plan with the best mean value after one opponent reply.

    Returns the chosen root plan (an action sequence ending in ``Pass()``, or a
    lethal that ends the game) — same shape as ``plan_turn``, so the policy plays
    it out identically.
    """
    seat = state.current
    opp_evaluator = opp_evaluator if opp_evaluator is not None else evaluator
    candidates = plan_turn_candidates(
        state, evaluator, width=width, max_actions=max_actions, k=n_plans
    )
    # Single candidate (e.g. forced line) — no reply search buys anything.
    if len(candidates) == 1:
        return candidates[0][1]

    # Simulate every (plan, world) leaf, accumulating terminal sentinels directly
    # and collecting the non-terminal leaves for ONE batched critic forward across
    # all plans/worlds (the same trunk-batching trick that keeps vbeam cheap). A
    # found lethal wins in every world, so it short-circuits without simulation.
    lethal = {i for i, (own_score, _p) in enumerate(candidates) if own_score >= _WIN_SCORE}
    plan_scores: list[list[float]] = [[] for _ in candidates]
    pending_views: list = []
    pending_ref: list[tuple[int, int]] = []  # (plan_index, sign)
    for pi, (_own_score, plan) in enumerate(candidates):
        if pi in lethal:
            continue
        for _ in range(n_worlds):
            det = determinize(state, rng, cards)
            score, view, sign = _simulate_to_leaf(
                det, plan, seat, opp_evaluator, width=width, max_actions=max_actions
            )
            if view is None:
                plan_scores[pi].append(score)
            else:
                pending_views.append(view)
                pending_ref.append((pi, sign))

    if pending_views:
        values = evaluator.values(pending_views)
        for (pi, sign), v in zip(pending_ref, values, strict=True):
            plan_scores[pi].append(sign * float(v))

    best_plan = None
    best_mean = -np.inf
    for pi, (_own_score, plan) in enumerate(candidates):
        mean = _WIN_SCORE if pi in lethal else sum(plan_scores[pi]) / len(plan_scores[pi])
        if mean > best_mean:
            best_mean, best_plan = mean, plan
    return best_plan


class RBeamBattlePolicy:
    """Battle policy that plans its turn with one opponent-reply ply, then plays it.

    Mirrors ``VBeamBattlePolicy``: the chosen root plan is computed once per turn
    and cached; subsequent within-turn calls pop the next action (the engine is
    deterministic within our own turn). Deterministic given the model and seed —
    the determinization RNG is reset per ``reset`` so replays are stable.
    """

    def __init__(
        self,
        model_path: str = "model.zip",
        name: str = "rbeam",
        width: int = 8,
        max_actions: int = 20,
        n_plans: int = 4,
        n_worlds: int = 4,
        seed: int = 0,
        evaluator=None,
        opp_evaluator=None,
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.width = width
        self.max_actions = max_actions
        self.n_plans = n_plans
        self.n_worlds = n_worlds
        self._seed = seed
        self._rng = random.Random(seed)
        self._evaluator = evaluator if evaluator is not None else NetValueEvaluator(model_path)
        # The opponent is modelled as an equally strong planner (same critic).
        self._opp_evaluator = opp_evaluator if opp_evaluator is not None else self._evaluator
        self._cards = load_cards()
        self._plan: list = []

    def reset(self, seed=None) -> None:
        self._rng = random.Random(self._seed if seed is None else seed)
        self._plan = []

    def battle_action(self, view, legal, state=None):
        if self._plan:
            a = self._plan.pop(0)
            if a in legal:
                return a
            self._plan = []  # stale cache (should not happen) — replan below
        if state is None:
            raise ValueError("RBeamBattlePolicy requires the forward-model `state` argument")
        if len(legal) == 1:
            return legal[0]
        plan = plan_turn_reply_aware(
            state,
            self._evaluator,
            cards=self._cards,
            rng=self._rng,
            width=self.width,
            max_actions=self.max_actions,
            n_plans=self.n_plans,
            n_worlds=self.n_worlds,
            opp_evaluator=self._opp_evaluator,
        )
        self._plan = plan[1:]
        return plan[0]
