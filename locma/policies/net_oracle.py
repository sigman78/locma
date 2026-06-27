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

import numpy as np

from locma.core.engine import make_battle_view
from locma.envs.encode import action_mask, encode_battle_tokens, sem_index


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

    # ------------------------------------------------------------------
    # Oracle interface
    # ------------------------------------------------------------------

    def priors(self, sim, actions: list, seat: int) -> list[float]:
        """Return masked policy priors aligned with ``actions``.

        Builds the ``BattleView`` from ``sim`` (which must be a ``GameState``),
        runs one forward pass through the net's actor head with the legal-action
        mask, then maps each action in ``actions`` to its semantic index and
        collects the corresponding probability.  The collected values are
        renormalised to sum to 1; if they sum to ~0 (net gave no mass to any
        legal action) the method falls back to uniform priors.

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
        import torch  # noqa: PLC0415 — lazy [ml] dep

        self._ensure()
        n = len(actions)
        if n == 0:
            return []

        view = make_battle_view(sim)
        obs = encode_battle_tokens(view)

        obs_t, _ = self._model.policy.obs_to_tensor(obs)
        mask = action_mask(view, actions)  # (155,) bool

        with torch.no_grad():
            dist = self._model.policy.get_distribution(obs_t, action_masks=mask)
            probs_t = dist.distribution.probs  # shape [1, 155]

        probs_np: np.ndarray = probs_t[0].cpu().numpy()  # shape [155]

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
        import torch  # noqa: PLC0415 — lazy [ml] dep

        self._ensure()

        view = make_battle_view(sim)
        obs = encode_battle_tokens(view)

        obs_t, _ = self._model.policy.obs_to_tensor(obs)

        with torch.no_grad():
            val_t = self._model.policy.predict_values(obs_t)  # shape [1, 1]

        v = float(val_t.item())

        # The critic speaks from sim.current's perspective; flip if needed
        if sim.current != root_seat:
            v = -v

        return max(-1.0, min(1.0, v))
