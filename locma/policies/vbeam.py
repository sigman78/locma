"""V-greedy own-turn beam planner ("planning-lite", E5 variant 1).

Attacks H2 (within-turn plan composition) with the minimum play-time compute:
at each decision point, beam-search *own-turn action sequences* on a cloned
forward-model state and score each stopping point with a learned value head
(the token PPO critic). The engine is deterministic and draws happen only at
``start_turn``, so own-turn lookahead uses no hidden information — the search
never simulates ``Pass`` (which would trigger the opponent's hidden draw) and
the evaluator reads only the public ``BattleView``. Fair by construction, like
``netdmcts``, but 10-100x cheaper (no determinization, no opponent model).

Sequences that end the game mid-turn score ±2.0 (outside the critic's [-1, 1]
clip range), so a found lethal always beats any value estimate and a forced
self-loss always loses to one. Everything is deterministic given the state and
the evaluator, so replays stay byte-identical.

The ML stack is only touched by ``NetValueEvaluator`` (lazy, inside methods);
``plan_turn`` and ``VBeamBattlePolicy`` with an injected evaluator are
import-safe without the [ml] extra.
"""

from __future__ import annotations

import numpy as np

from locma.core import battle as battlemod
from locma.core.actions import Pass
from locma.core.engine import make_battle_view
from locma.core.state import Phase
from locma.envs.encode import N_TACTICAL_V1, encode_battle_tokens
from locma.policies.mcts import _clone_battle

# Terminal scores sit outside the critic's [-1, 1] clip range so a real win /
# loss always dominates any value estimate.
_WIN_SCORE = 2.0
_LOSS_SCORE = -2.0


def plan_turn(state, evaluator, *, width: int = 8, max_actions: int = 20) -> list:
    """Beam-search own-turn action sequences; return the best complete plan.

    Parameters
    ----------
    state:
        A ``GameState`` in the BATTLE phase at the planner's decision point.
        Never mutated (cloned via ``_clone_battle`` before any apply).
    evaluator:
        An object exposing ``values(views: list[BattleView]) -> list[float]``,
        each value in [-1, 1] from the view owner's perspective.
    width:
        Beam width — non-terminal candidates kept per depth.
    max_actions:
        Safety cap on plan length (a LOCM turn is at most ~14 atomic actions;
        the beam normally exhausts its legal continuations first).

    Returns
    -------
    list
        The action sequence to play, ending with ``Pass()`` unless the plan
        wins the game outright (the engine ends the episode, so no Pass is
        needed or possible). Never empty: the "stop now" plan ``[Pass()]`` is
        always a candidate, so a plan exists even when every action looks bad.

    Notes
    -----
    Every explored state is a stopping candidate scored ``V(state)``; along
    the net's own preferred line these are near-ties (V is approximately
    conserved under its own optimal actions), so the search adds value exactly
    where reactive play loses it: multi-step compositions whose payoff only
    shows after a locally neutral first step (kill the Guard with the weak
    attacker so the strong one goes face). Ties break toward the earliest
    (shortest) candidate, so the planner never pads a plan with actions the
    value head is indifferent to.
    """
    seat = state.current
    root = _clone_battle(state)
    root_view = make_battle_view(root)

    # Completed plans: (score, insertion_order, plan). Order breaks score ties
    # deterministically toward the earliest-found (shortest) plan. The root
    # "stop now" plan is always present so the result is never empty.
    completed: list[tuple[float, int, list]] = [
        (float(evaluator.values([root_view])[0]), 0, [Pass()])
    ]
    order = 1

    beam: list[tuple[object, list]] = [(root, [])]
    seen = {root_view}  # collapse action-order permutations that meet again

    for _depth in range(max_actions):
        sims: list = []
        views: list = []
        plans: list[list] = []
        for sim, plan in beam:
            for a in battlemod.battle_legal(sim):
                if isinstance(a, Pass):
                    continue  # Pass = stop; stopping is scored via `completed`
                s2 = _clone_battle(sim)
                battlemod.apply_battle(s2, a)
                plan2 = [*plan, a]
                if s2.phase == Phase.ENDED:
                    if s2.winner == seat:
                        return plan2  # nothing can beat an immediate win
                    completed.append((_LOSS_SCORE, order, plan2))
                    order += 1
                    continue
                v2 = make_battle_view(s2)
                if v2 in seen:
                    continue
                seen.add(v2)
                sims.append(s2)
                views.append(v2)
                plans.append(plan2)
        if not sims:
            break

        vals = evaluator.values(views)
        ranked = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
        for i in ranked:
            completed.append((float(vals[i]), order, [*plans[i], Pass()]))
            order += 1
        beam = [(sims[i], plans[i]) for i in ranked[:width]]

    best = max(completed, key=lambda t: (t[0], -t[1]))
    return best[2]


class NetValueEvaluator:
    """Batched value-head evaluator over a token ``MaskablePPO`` model.

    Loads the model lazily on first use (mirrors ``NetOracle``), forces eval
    mode (the extractor has dropout), and detects the token obs variant from
    the model's observation space. ``values`` runs ONE batched trunk forward
    for any number of views — this is what keeps the beam cheap.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        self._variant = "v0"

    def _ensure(self) -> None:
        if self._model is None:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

            self._model = MaskablePPO.load(self.model_path)
            self._model.policy.set_training_mode(False)
            import gymnasium  # noqa: PLC0415 — lazy, only reached after model load

            if not isinstance(self._model.observation_space, gymnasium.spaces.Dict):
                raise ValueError(
                    "NetValueEvaluator requires a token (Dict-obs) model; "
                    f"got {type(self._model.observation_space).__name__}"
                )
            n_scalar = int(self._model.observation_space["scalars"].shape[0])
            self._variant = "v1" if n_scalar == N_TACTICAL_V1 else "v0"

    def values(self, views: list) -> list[float]:
        """Critic values in [-1, 1] for each view, from the view owner's side."""
        import torch  # noqa: PLC0415 — lazy [ml] dep

        self._ensure()
        obs_list = [encode_battle_tokens(v, self._variant) for v in views]
        batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

        policy = self._model.policy
        obs_t, _ = policy.obs_to_tensor(batch)
        with torch.no_grad():
            features = policy.extract_features(obs_t)
            _, latent_vf = policy.mlp_extractor(features)
            raw = policy.value_net(latent_vf).squeeze(-1).cpu().numpy()
        return [max(-1.0, min(1.0, float(x))) for x in np.atleast_1d(raw)]


class VBeamBattlePolicy:
    """Battle policy that plans its whole turn with ``plan_turn`` and plays it out.

    The plan is computed once per turn and cached; subsequent calls within the
    same turn pop the next planned action (the engine is deterministic within
    the own turn, so the cached tail stays valid). A legality guard replans if
    the cached action is somehow stale — belt-and-suspenders, it should never
    fire. Deterministic given the model, so replays are stable.
    """

    def __init__(
        self,
        model_path: str = "model.zip",
        name: str = "vbeam",
        width: int = 8,
        max_actions: int = 20,
        evaluator=None,
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.width = width
        self.max_actions = max_actions
        self._evaluator = evaluator if evaluator is not None else NetValueEvaluator(model_path)
        self._plan: list = []

    def reset(self, seed=None) -> None:
        self._plan = []

    def battle_action(self, view, legal, state=None):
        if self._plan:
            a = self._plan.pop(0)
            if a in legal:
                return a
            self._plan = []  # stale cache (should not happen) — replan below
        if state is None:
            raise ValueError("VBeamBattlePolicy requires the forward-model `state` argument")
        if len(legal) == 1:
            return legal[0]
        plan = plan_turn(state, self._evaluator, width=self.width, max_actions=self.max_actions)
        self._plan = plan[1:]
        return plan[0]
