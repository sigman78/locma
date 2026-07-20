"""NetOracle: masked policy priors + value from a token MaskablePPO net.

Reads only the fair ``BattleView`` (no hidden information) and wraps a
lazily-loaded token ``MaskablePPO`` model as a PUCT oracle.

Oracle protocol (injected into ``puct.puct_search``)::

    oracle.priors(sim, actions, seat) -> list[float]
    oracle.value(sim, root_seat)      -> float  # in [-1, 1]

The model is NOT loaded until the first call, mirroring
``MaskablePPOBattlePolicy._ensure``. Torch/sb3 imports stay inside methods
so this module is import-safe without the [ml] extra.
"""

from __future__ import annotations

import random

import numpy as np

from locma.core import battle as battlemod
from locma.core.engine import make_battle_view
from locma.data.cards_db import load_cards
from locma.envs.encode import (
    action_mask,
    encode_battle,
    encode_battle_tokens,
    sem_index,
    token_variant_for_space,
)
from locma.policies.mcts import determinize
from locma.policies.puct import puct_search


class NetOracle:
    """PUCT oracle backed by a token ``MaskablePPO`` net.

    Parameters
    ----------
    model_path:
        Path to a saved ``MaskablePPO`` ``.zip`` file.  The model is loaded
        lazily on the first ``priors`` or ``value`` call.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        # 1-entry cache: (sim_object, raw_value) from the last priors() call.
        # value() reuses this when called on the same sim (compare with `is`).
        self._value_cache: tuple | None = None

    # ------------------------------------------------------------------
    # Lazy loading (mirrors MaskablePPOBattlePolicy._ensure)
    # ------------------------------------------------------------------

    def _ensure(self) -> None:
        if self._model is None:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

            self._model = MaskablePPO.load(self.model_path)
            # MaskablePPO.load() does NOT set eval mode; the transformer extractor
            # has dropout=0.1, so training mode makes every forward pass
            # non-deterministic.  Force eval here once and keep it.
            self._model.policy.set_training_mode(False)
            # Guard: only token (Dict-obs) models are supported — a flat Box-obs
            # model would fail later with a cryptic shape error inside obs_to_tensor.
            import gymnasium  # noqa: PLC0415 — lazy, only reached after model load

            if not isinstance(self._model.observation_space, gymnasium.spaces.Dict):
                raise ValueError(
                    "NetOracle requires a token (Dict-obs) model; "
                    f"got {type(self._model.observation_space).__name__}"
                )

    # ------------------------------------------------------------------
    # Single combined forward (P10 optimisation)
    # ------------------------------------------------------------------

    def _forward(self, view, mask):
        """Run ONE trunk forward pass; return ``(probs_np, raw_value)``.

        Parameters
        ----------
        view:
            A ``BattleView`` already built from the ``sim`` of interest.
        mask:
            Boolean action-mask array of shape ``(155,)``, or ``None``.
            When ``None`` the policy head is skipped and ``probs_np`` is
            returned as ``None`` (value-only path).

        Returns
        -------
        tuple[np.ndarray | None, float]
            ``probs_np`` is ``None`` when ``mask`` is ``None``; otherwise a
            ``(155,)`` float32 array of **masked** action probabilities.
            ``raw_value`` is the critic output from ``sim.current``'s
            perspective (pre-sign-flip, pre-clip).
        """
        import torch  # noqa: PLC0415 — lazy [ml] dep

        self._ensure()
        obs = encode_battle_tokens(view, token_variant_for_space(self._model.observation_space))
        obs_t, _ = self._model.policy.obs_to_tensor(obs)

        with torch.no_grad():
            features = self._model.policy.extract_features(obs_t)
            latent_pi, latent_vf = self._model.policy.mlp_extractor(features)
            raw_value = float(self._model.policy.value_net(latent_vf).item())
            if mask is None:
                return None, raw_value
            dist = self._model.policy._get_action_dist_from_latent(latent_pi)
            dist.apply_masking(mask)
            probs_np: np.ndarray = dist.distribution.probs[0].cpu().numpy()

        return probs_np, raw_value

    # ------------------------------------------------------------------
    # Oracle interface
    # ------------------------------------------------------------------

    def priors(self, sim, actions: list, seat: int) -> list[float]:
        """Return masked policy priors aligned with ``actions``.

        Builds the ``BattleView`` from ``sim`` (which must be a ``GameState``),
        runs one combined forward pass through the trunk (computing BOTH the
        policy probs and the critic value in a single pass), stashes the raw
        value in ``self._value_cache`` for reuse by the subsequent ``value()``
        call on the same ``sim``, then maps each action in ``actions`` to its
        semantic index and collects the corresponding probability.  The
        collected values are renormalised to sum to 1; if they sum to ~0 (net
        gave no mass to any legal action) the method falls back to uniform
        priors.

        Parameters
        ----------
        sim:
            A ``GameState`` in the BATTLE phase (read-only).
        actions:
            The legal actions at this node (``battle_legal(sim)``).
        seat:
            The seat to move (used only for the oracle protocol; the view is
            already from ``sim.current``'s perspective).

        Returns
        -------
        list[float]
            Prior probabilities aligned with ``actions``, summing to 1.
        """
        n = len(actions)
        if n == 0:
            return []

        view = make_battle_view(sim)
        mask = action_mask(view, actions)  # (155,) bool

        probs_np, raw_value = self._forward(view, mask)

        # Stash raw value (from sim.current's perspective) for value() reuse
        self._value_cache = (sim, raw_value)

        collected = []
        for a in actions:
            idx = sem_index(view, a)
            collected.append(float(probs_np[idx]) if idx is not None else 0.0)

        total = sum(collected)
        if total < 1e-9:
            # Net gave zero mass to every legal action — fall back to uniform
            return [1.0 / n] * n
        return [x / total for x in collected]

    def value(self, sim, root_seat: int) -> float:
        """Return the net's critic value for ``sim``, from ``root_seat``'s perspective.

        When ``priors()`` was called just before on the **same** ``sim`` object,
        the raw critic value is reused from the cache (no second trunk pass).
        Otherwise a standalone forward is run (value-only; cheaper than a full
        combined forward since the policy head is skipped).

        The critic evaluates from ``sim.current``'s perspective; if
        ``sim.current != root_seat`` the sign is flipped.  The result is
        clipped to [-1.0, 1.0] (the net's value head is unbounded).

        Parameters
        ----------
        sim:
            A ``GameState`` in the BATTLE phase (read-only).
        root_seat:
            The seat from whose perspective the value should be expressed
            (the PUCT root seat).

        Returns
        -------
        float
            Value estimate in [-1.0, 1.0].
        """
        if self._value_cache is not None and self._value_cache[0] is sim:
            # Cache hit: reuse the raw value from the preceding priors() call
            raw = self._value_cache[1]
        else:
            # Cache miss: standalone value-only forward (skips policy head)
            _, raw = self._forward(make_battle_view(sim), None)

        # The critic speaks from sim.current's perspective; flip if needed
        v = raw if sim.current == root_seat else -raw
        return max(-1.0, min(1.0, v))


class NetGuidedDMCTSBattlePolicy:
    """Fair net-guided determinized PUCT (Phase-1 netdmcts).

    Like ``DMCTSBattlePolicy`` but uses a ``NetOracle`` (net policy priors + critic
    value) in place of the heuristic rollout.  Samples ``determinizations`` plausible
    hidden worlds, runs ``puct_search`` with the net oracle on each, accumulates the
    root edge visit counts across all worlds, and picks the action with the most total
    visits.

    Root legal actions are identical across determinized worlds: ``determinize`` keeps
    the agent's own hand and board real, only resampling the opponent's hidden hand/deck,
    so ``battle_legal`` is the same in every world.

    ``deterministic=True`` seeds the sampling and search from a hash of the public
    observation so the move is a stable function of the ``BattleView`` (mirrors
    ``DMCTSBattlePolicy``'s deterministic distillation path).
    """

    def __init__(
        self,
        name: str = "netdmcts",
        model_path: str = "model.zip",
        determinizations: int = 15,
        iterations: int = 80,
        c_puct: float = 1.5,
        seed: int = 0,
        deterministic: bool = False,
    ) -> None:
        if determinizations < 1 or iterations < 1:
            raise ValueError("determinizations and iterations must be >= 1")
        self.name = name
        self.model_path = model_path
        self.K = determinizations
        self.iterations = iterations
        self.c_puct = c_puct
        self.deterministic = deterministic
        self._seed = seed
        self._r = random.Random(seed)
        self._cards = load_cards()
        self._oracle = NetOracle(model_path)

    def reset(self, seed=None) -> None:
        """Reseed the internal RNG (mirrors ``DMCTSBattlePolicy.reset``)."""
        s = self._seed if seed is None else seed
        self._r = random.Random(s)

    def battle_action(self, view, legal, state=None):
        """Return a legal action using net-guided determinized PUCT.

        Parameters
        ----------
        view:
            The public ``BattleView`` for the current player.
        legal:
            The list of legal actions at this state.
        state:
            The live ``GameState`` forward-model (required).

        Raises
        ------
        ValueError
            If ``state`` is ``None``.
        """
        if state is None:
            raise ValueError(
                "NetGuidedDMCTSBattlePolicy requires the forward-model `state` argument"
            )
        if len(legal) == 1:
            return legal[0]

        # Deterministic distillation path: seed RNG from the observation hash
        if self.deterministic:
            import hashlib  # noqa: PLC0415 — only the distillation path needs this

            obs = encode_battle(view)
            seed = int.from_bytes(hashlib.blake2b(obs.tobytes(), digest_size=7).digest(), "little")
            rng = random.Random(seed)
        else:
            rng = self._r

        # Accumulate root edge visit counts across K determinized worlds
        total: list[int] | None = None
        for _ in range(self.K):
            det = determinize(state, rng, self._cards)
            counts = puct_search(det, self._oracle, self.iterations, self.c_puct, rng)
            if total is None:
                total = list(counts)
            else:
                for i in range(len(total)):
                    total[i] += counts[i]

        # Map back to the real state's legal actions
        actions = list(battlemod.battle_legal(state))
        assert len(actions) == len(total), (
            f"action/count length mismatch: {len(actions)} vs {len(total)}"
        )
        best = max(range(len(total)), key=lambda i: total[i])
        return actions[best]
