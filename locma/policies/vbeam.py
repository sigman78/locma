"""V-greedy own-turn beam planner ("planning-lite", E5 variant 1).

Attacks H2 (within-turn plan composition) with the minimum play-time compute:
at each decision point, beam-search *own-turn action sequences* on a cloned
forward-model state and score stopping points with a learned value head (the
token PPO critic). The engine is deterministic and draws happen only at
``start_turn``, so own-turn lookahead uses no hidden information — the search
never simulates ``Pass`` (which would trigger the opponent's hidden draw) and
the evaluator reads only the public ``BattleView``. Fair by construction, like
``netdmcts``, but 10-100x cheaper (no determinization, no opponent model).

Stop-scoring rule (the load-bearing subtlety): V(s) is a *state* value — it
already credits the actions the net expects to take from s. Scoring "stop
here" with V(s) therefore free-rides on actions the plan never takes; the
naive version passed 1/3 of its turns and regressed -0.34 avg-hard3 against
its own reactive net. A stopping point is only scoreable with V(s) where that
bias vanishes: when the net's own masked argmax at s IS Pass (continuing ==
passing, so V(s) contains no phantom actions) or when Pass is the only legal
action (the line is exhausted). Everywhere else "stop now" is a last-resort
fallback ranked above self-inflicted losses only.

Sequences that end the game mid-turn score outside the critic's [-1, 1] clip
range (win +2, loss -2, fallback stop -1.5), so a found lethal always beats
any value estimate and actively losing always ranks below passing.

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
from locma.envs.encode import N_TACTICAL_V1, action_mask, encode_battle_tokens
from locma.policies.mcts import _clone_battle

# Scores outside the critic's [-1, 1] clip range: a real win/loss always
# dominates any value estimate, and the net-disapproved root fallback ranks
# above actively losing but below every legitimate stopping point.
_WIN_SCORE = 2.0
_LOSS_SCORE = -2.0
_FALLBACK_STOP_SCORE = -1.5


def plan_turn(state, evaluator, *, width: int = 8, max_actions: int = 20, collect=None) -> list:
    """Beam-search own-turn action sequences; return the best complete plan.

    Parameters
    ----------
    state:
        A ``GameState`` in the BATTLE phase at the planner's decision point.
        Never mutated (cloned via ``_clone_battle`` before any apply).
    evaluator:
        An object exposing
        ``evaluate(views, masks) -> (values, would_pass)`` where ``values``
        are critic estimates in [-1, 1] from the view owner's perspective and
        ``would_pass[i]`` is True when the policy's masked argmax at view i is
        Pass (the condition under which ``values[i]`` is a bias-free stop
        score — see module docstring).
    width:
        Beam width — non-terminal candidates kept per depth.
    max_actions:
        Safety cap on plan length (a LOCM turn is at most ~14 atomic actions;
        the beam normally exhausts its legal continuations first).
    collect:
        Optional list for AZ-style backed-up value targets (E5 variant 2b).
        When given, the search appends one ``(view, target, depth, stop_ok)``
        tuple per explored state whose subtree produced at least one completed
        plan: ``target`` = the best completed-plan score reachable through
        that state, clipped to [-1, 1] (so searched wins/losses ground it),
        which DIFFERS between sibling states — unlike Monte-Carlo game labels.
        ``None`` (default) changes nothing.

    Returns
    -------
    list
        The action sequence to play, ending with ``Pass()`` unless the plan
        wins the game outright (the engine ends the episode, so no Pass is
        needed or possible). Never empty: the root "stop now" plan is always
        present, as a first-class candidate when the net itself would pass at
        the root, else as the ranked-last fallback.
    """
    seat = state.current
    root = _clone_battle(state)
    root_view = make_battle_view(root)
    root_legal = list(battlemod.battle_legal(root))
    root_mask = action_mask(root_view, root_legal)

    vals, would_pass = evaluator.evaluate([root_view], [root_mask])
    root_stop_ok = would_pass[0] or len(root_legal) == 1

    # Completed plans: (score, insertion_order, plan). Order breaks score ties
    # deterministically toward the earliest-found (shortest) plan.
    completed: list[tuple[float, int, list]] = [
        (float(vals[0]) if root_stop_ok else _FALLBACK_STOP_SCORE, 0, [Pass()])
    ]
    order = 1

    beam: list[tuple[object, list, list]] = [(root, root_legal, [])]
    seen = {root_view}  # collapse action-order permutations that meet again
    explored = [(root_view, (), 0, root_stop_ok)] if collect is not None else None
    win_found = False

    for _depth in range(max_actions):
        sims: list = []
        views: list = []
        masks: list = []
        legals: list[list] = []
        plans: list[list] = []
        for sim, legal, plan in beam:
            for a in legal:
                if isinstance(a, Pass):
                    continue  # Pass = stop; stopping is scored via `completed`
                s2 = _clone_battle(sim)
                battlemod.apply_battle(s2, a)
                plan2 = [*plan, a]
                if s2.phase == Phase.ENDED:
                    if s2.winner == seat:
                        # Nothing can beat an immediate win: record and stop.
                        completed.append((_WIN_SCORE, order, plan2))
                        order += 1
                        win_found = True
                        break
                    completed.append((_LOSS_SCORE, order, plan2))
                    order += 1
                    continue
                v2 = make_battle_view(s2)
                if v2 in seen:
                    continue
                seen.add(v2)
                l2 = list(battlemod.battle_legal(s2))
                sims.append(s2)
                views.append(v2)
                masks.append(action_mask(v2, l2))
                legals.append(l2)
                plans.append(plan2)
            if win_found:
                break
        if win_found or not sims:
            break

        vals, would_pass = evaluator.evaluate(views, masks)
        ranked = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
        for i in ranked:
            stop_ok = would_pass[i] or len(legals[i]) == 1
            if explored is not None:
                explored.append((views[i], tuple(plans[i]), _depth + 1, stop_ok))
            if stop_ok:
                completed.append((float(vals[i]), order, [*plans[i], Pass()]))
                order += 1
        beam = [(sims[i], legals[i], plans[i]) for i in ranked[:width]]

    if collect is not None:
        _harvest_backups(collect, explored, completed)
    best = max(completed, key=lambda t: (t[0], -t[1]))
    return best[2]


def _harvest_backups(collect: list, explored: list, completed: list) -> None:
    """Map each explored state to its backed-up target (E5 variant 2b).

    A state's target is the best completed-plan score whose action sequence
    extends the state's prefix, clipped to [-1, 1] (win +2 -> +1, loss
    -2 -> -1, critic stops as-is). The net-disapproved root fallback is
    excluded — it is an artificial sentinel, not an achievable value. States
    whose subtree completed no plan (beam-pruned before any stop) get no
    sample.
    """
    best_ext: dict[tuple, float] = {}
    root_target = None
    for score, _order, plan in completed:
        if score == _FALLBACK_STOP_SCORE:
            continue
        t = max(-1.0, min(1.0, float(score)))
        root_target = t if root_target is None else max(root_target, t)
        actions = tuple(a for a in plan if not isinstance(a, Pass))
        for k in range(1, len(actions) + 1):
            p = actions[:k]
            if t > best_ext.get(p, -2.0):
                best_ext[p] = t
    for view, prefix, depth, stop_ok in explored:
        t = root_target if prefix == () else best_ext.get(prefix)
        if t is not None:
            collect.append((view, t, depth, stop_ok))


class NetValueEvaluator:
    """Batched critic + pass-preference evaluator over a token ``MaskablePPO``.

    Loads the model lazily on first use (mirrors ``NetOracle``), forces eval
    mode (the extractor has dropout), and detects the token obs variant from
    the model's observation space. ``evaluate`` runs ONE batched trunk forward
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

    def _forward(self, views: list, masks: list | None):
        """One batched trunk pass; returns (raw_values, probs_or_None)."""
        import torch  # noqa: PLC0415 — lazy [ml] dep

        self._ensure()
        obs_list = [encode_battle_tokens(v, self._variant) for v in views]
        batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

        policy = self._model.policy
        obs_t, _ = policy.obs_to_tensor(batch)
        with torch.no_grad():
            features = policy.extract_features(obs_t)
            latent_pi, latent_vf = policy.mlp_extractor(features)
            raw = policy.value_net(latent_vf).squeeze(-1).cpu().numpy()
            if masks is None:
                return np.atleast_1d(raw), None
            dist = policy._get_action_dist_from_latent(latent_pi)
            dist.apply_masking(np.stack(masks))
            probs = dist.distribution.probs.cpu().numpy()  # (B, 155)
        return np.atleast_1d(raw), probs

    def evaluate(self, views: list, masks: list) -> tuple[list[float], list[bool]]:
        """Critic values in [-1, 1] + whether the policy's masked argmax is Pass.

        Pass occupies semantic action index 0 (see ``encode.sem_index``), so
        ``would_pass[i]`` is simply ``argmax(probs[i]) == 0``.
        """
        raw, probs = self._forward(views, masks)
        values = [max(-1.0, min(1.0, float(x))) for x in raw]
        would_pass = [int(np.argmax(p)) == 0 for p in probs]
        return values, would_pass

    def values(self, views: list) -> list[float]:
        """Critic values only (no policy head) — used by equivalence tests."""
        raw, _ = self._forward(views, None)
        return [max(-1.0, min(1.0, float(x))) for x in raw]


class EnsembleValueEvaluator:
    """Mean-of-critics evaluator over several token models (zero-training trio).

    ``values[i]`` is the mean of each member's *clipped* critic estimate (each
    member contributes exactly what it would contribute alone), and
    ``would_pass[i]`` is the argmax-is-Pass test on the mean of the members'
    masked policy distributions — the standard probability-averaging ensemble.
    Members load lazily like ``NetValueEvaluator``; cost is one trunk forward
    per member per beam depth.
    """

    def __init__(self, model_paths: list[str]) -> None:
        if len(model_paths) < 2:
            raise ValueError("EnsembleValueEvaluator needs at least 2 model paths")
        self.model_paths = list(model_paths)
        self.members = [NetValueEvaluator(p) for p in model_paths]

    def evaluate(self, views: list, masks: list) -> tuple[list[float], list[bool]]:
        raws, probss = zip(*(m._forward(views, masks) for m in self.members), strict=True)
        values_mat = np.clip(np.stack(raws), -1.0, 1.0)
        values = [float(x) for x in values_mat.mean(axis=0)]
        mean_probs = np.stack(probss).mean(axis=0)
        would_pass = [int(np.argmax(p)) == 0 for p in mean_probs]
        return values, would_pass

    def values(self, views: list) -> list[float]:
        raws = [np.clip(m._forward(views, None)[0], -1.0, 1.0) for m in self.members]
        return [float(x) for x in np.stack(raws).mean(axis=0)]


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
        collect=None,
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.width = width
        self.max_actions = max_actions
        self._evaluator = evaluator if evaluator is not None else NetValueEvaluator(model_path)
        self._plan: list = []
        # Optional backed-up-target sink, forwarded to plan_turn (E5 variant 2b).
        self.collect = collect

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
        plan = plan_turn(
            state,
            self._evaluator,
            width=self.width,
            max_actions=self.max_actions,
            collect=self.collect,
        )
        self._plan = plan[1:]
        return plan[0]
