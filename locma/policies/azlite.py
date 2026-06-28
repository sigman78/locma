"""AlphaZero-lite: PUCT-guided MCTS with heuristic or learned policy priors."""

from __future__ import annotations

import math
import random
from collections import Counter

from locma.core import battle as battlemod
from locma.core.state import Phase
from locma.envs.encode import action_mask, encode_battle, sem_index
from locma.policies.battles import RandomBattlePolicy
from locma.policies.mcts import DMCTSBattlePolicy, _board_power, _clone_battle
from locma.policies.puct import _reward as _puct_reward
from locma.policies.puct import puct_search


def _leaf_value(sim, seat: int) -> float:
    """Board/health heuristic in [-1, 1] from ``seat``'s perspective."""
    me = sim.players[seat]
    op = sim.players[1 - seat]
    h = (me.health - op.health) / 30.0
    b = (_board_power(me) - _board_power(op)) / 20.0
    v = h + 0.5 * b
    return 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)


class _HeuristicOracle:
    """Thin adapter exposing AZLiteBattlePolicy's prior/value as the PUCT oracle."""

    def __init__(self, policy: AZLiteBattlePolicy):
        self._p = policy

    def priors(self, sim, actions: list, seat: int) -> list[float]:
        return self._p._prior(sim, actions, seat)

    def value(self, sim, root_seat: int) -> float:
        return self._p._value(sim, root_seat)


class AZLiteBattlePolicy:
    def __init__(
        self,
        name: str = "azlite",
        iterations: int = 100,
        c_puct: float = 1.5,
        seed: int = 0,
        rollout_turns: int = 0,
        tau: float = 0.4,
        turn_cap: int = 200,
    ):
        self.name = name
        self.iterations = iterations
        self.c_puct = c_puct
        self._seed = seed
        self.rollout_turns = rollout_turns
        self.tau = tau
        self.turn_cap = turn_cap
        self._rollout = RandomBattlePolicy("azlite-rollout", seed=seed)
        self._r = random.Random(seed)

    def reset(self, seed=None):
        s = self._seed if seed is None else seed
        self._r = random.Random(s)
        self._rollout.reset(s)

    def _prior(self, sim, actions, seat: int) -> list[float]:
        """1-ply heuristic lookahead softmax: how good each action looks for `seat`."""
        if len(actions) == 1:
            return [1.0]
        vals = []
        for a in actions:
            s2 = _clone_battle(sim)
            battlemod.apply_battle(s2, a)
            vals.append(_leaf_value(s2, seat))
        m = max(vals)
        exps = [math.exp((v - m) / self.tau) for v in vals]
        z = sum(exps)
        return [e / z for e in exps]

    def _reward(self, sim, root_seat: int) -> float:
        return _puct_reward(sim, root_seat)

    def _value(self, sim, root_seat: int) -> float:
        if self.rollout_turns <= 0:
            return _leaf_value(sim, root_seat)
        tc = 0
        while sim.phase == Phase.BATTLE and sim.turn <= self.turn_cap and tc < self.rollout_turns:
            owner = sim.current
            legal = battlemod.battle_legal(sim)
            battlemod.apply_battle(sim, self._rollout.battle_action(None, legal))
            if sim.current != owner:
                tc += 1
        if sim.phase != Phase.BATTLE:
            return self._reward(sim, root_seat)
        return _leaf_value(sim, root_seat)

    def battle_action(self, view, legal, state=None):
        if state is None:
            raise ValueError("AZLiteBattlePolicy requires the forward-model `state` argument")
        if len(legal) == 1:
            return legal[0]

        oracle = _HeuristicOracle(self)
        visit_counts = puct_search(state, oracle, self.iterations, self.c_puct, self._r)
        root_actions = list(battlemod.battle_legal(state))
        best = max(range(len(root_actions)), key=lambda i: visit_counts[i])
        return root_actions[best]


class PUCTPPOBattlePolicy(AZLiteBattlePolicy):
    """PUCT search with PPO policy-head priors and the AZ-lite heuristic value."""

    def __init__(
        self,
        model_path: str = "model.zip",
        name: str = "puct-ppo",
        iterations: int = 100,
        c_puct: float = 1.5,
        seed: int = 0,
        rollout_turns: int = 0,
        obs_mode: str = "auto",
        turn_cap: int = 200,
    ):
        super().__init__(
            name=name,
            iterations=iterations,
            c_puct=c_puct,
            seed=seed,
            rollout_turns=rollout_turns,
            turn_cap=turn_cap,
        )
        self.model_path = model_path
        self.obs_mode = "flat" if obs_mode == "base" else obs_mode
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415

            self._model = MaskablePPO.load(self.model_path)

    def _encode_obs(self, view, actions):
        if self.obs_mode == "tactical":
            from locma.envs.encode_tactical import encode_battle as encode_tactical  # noqa: PLC0415

            return encode_tactical(view, actions)
        if self.obs_mode in {"auto", "flat"}:
            from gymnasium import spaces  # noqa: PLC0415

            observation_space = getattr(self._model, "observation_space", None)
            if isinstance(observation_space, spaces.Dict):
                from locma.envs.encode import encode_battle_tokens  # noqa: PLC0415

                return encode_battle_tokens(view)
            return encode_battle(view)
        raise ValueError(f"unknown obs_mode {self.obs_mode!r}")

    def _prior(self, sim, actions, seat: int) -> list[float]:
        if len(actions) == 1:
            return [1.0]

        from locma.core.engine import make_battle_view  # noqa: PLC0415

        self._ensure_model()
        view = make_battle_view(sim)
        mask = action_mask(view, actions)
        obs = self._encode_obs(view, actions)

        import torch as th  # noqa: PLC0415

        obs_tensor, _ = self._model.policy.obs_to_tensor(obs)
        with th.no_grad():
            dist = self._model.policy.get_distribution(obs_tensor, action_masks=mask[None, :])
            probs = dist.distribution.probs.detach().cpu().numpy()[0]

        priors = [
            float(probs[idx]) if idx is not None else 0.0
            for idx in (sem_index(view, a) for a in actions)
        ]
        z = sum(priors)
        if z <= 0.0:
            return [1.0 / len(actions)] * len(actions)
        return [p / z for p in priors]


class DeterminizedPUCTPPOBattlePolicy:
    """Fair imperfect-information wrapper around PPO-prior PUCT."""

    def __init__(
        self,
        model_path: str = "model.zip",
        name: str = "dpuct-ppo",
        determinizations: int = 5,
        iterations: int = 5,
        c_puct: float = 1.5,
        seed: int = 0,
        rollout_turns: int = 0,
        obs_mode: str = "auto",
    ):
        self.name = name
        self.K = determinizations
        self.I = iterations
        self.c_puct = c_puct
        self._seed = seed
        self.rollout_turns = rollout_turns
        self.obs_mode = obs_mode
        self._r = random.Random(seed)
        self._determinizer = DMCTSBattlePolicy(seed=seed, reshuffle_own=True)
        self._inner = PUCTPPOBattlePolicy(
            model_path=model_path,
            iterations=iterations,
            c_puct=c_puct,
            seed=seed,
            rollout_turns=rollout_turns,
            obs_mode=obs_mode,
        )

    @property
    def model_path(self) -> str:
        return self._inner.model_path

    def reset(self, seed=None):
        s = self._seed if seed is None else seed
        self._r = random.Random(s)
        self._determinizer.reset(s)
        self._inner.reset(s)

    def battle_action(self, view, legal, state=None):
        if state is None:
            raise ValueError("DeterminizedPUCTPPOBattlePolicy requires the forward-model `state`")
        if len(legal) == 1:
            return legal[0]

        votes: Counter = Counter()
        for _ in range(self.K):
            det = self._determinizer._determinize(state, self._r)
            votes[self._inner.battle_action(view, legal, det)] += 1
        return votes.most_common(1)[0][0]
