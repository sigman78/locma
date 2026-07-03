"""MaskablePPO-backed battle policy with lazy model loading (requires [ml])."""

from __future__ import annotations

import numpy as np

from locma.envs.encode import action_mask, encode_battle, encode_battle_tokens, index_to_action


def _encode_for(model, view):
    """Select the observation encoder based on the loaded model's observation space.

    ``from gymnasium import spaces`` is kept inside the function body so that this
    module remains import-safe without the [ml] stack — gymnasium is only available
    once a model has been loaded.
    """
    from gymnasium import spaces  # noqa: PLC0415 — lazy, only reached after model load

    if isinstance(model.observation_space, spaces.Dict):
        n_scalar = int(model.observation_space["scalars"].shape[0])
        variant = "v1" if n_scalar == 18 else "v0"
        return encode_battle_tokens(view, variant)
    return encode_battle(view)


class MaskablePPOBattlePolicy:
    """Wraps a saved MaskablePPO model as a Battle Policy.

    The model is NOT loaded until the first ``battle_action`` call, so
    construction works even when the file does not yet exist and never imports
    the ``[ml]`` stack. ``deterministic`` stays True for byte-identical replay.
    """

    def __init__(
        self,
        model_path: str = "model.zip",
        name: str = "ppo",
        deterministic: bool = True,
        model=None,
    ):
        self.model_path = model_path
        self.name = name
        self.deterministic = deterministic
        self._model = model  # if provided, skip the lazy file load

    def _ensure(self) -> None:
        if self._model is None:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

            self._model = MaskablePPO.load(self.model_path)

    def battle_action(self, view, legal, state=None):
        self._ensure()
        obs = _encode_for(self._model, view)
        mask = action_mask(view, legal)
        idx, _ = self._model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return index_to_action(view, legal, int(idx))

    def reset(self, seed=None) -> None:
        pass


class RecurrentPPOBattlePolicy:
    """Wraps a saved MaskableRecurrentPPO model as a stateful Battle Policy.

    The LSTM hidden state is carried across all of one game's decisions and
    cleared by ``reset()`` (the harness calls it once per game), so the net
    sees the same within-episode state flow it was trained on. Lazy model
    loading and ``deterministic=True`` mirror MaskablePPOBattlePolicy.
    """

    def __init__(
        self,
        model_path: str = "model.zip",
        name: str = "rppo",
        deterministic: bool = True,
        model=None,
    ):
        self.model_path = model_path
        self.name = name
        self.deterministic = deterministic
        self._model = model
        self._state = None
        self._episode_start = True

    def _ensure(self) -> None:
        if self._model is None:
            from locma.envs.rppo import MaskableRecurrentPPO  # noqa: PLC0415 — optional [ml] dep

            self._model = MaskableRecurrentPPO.load(self.model_path)

    def battle_action(self, view, legal, state=None):
        self._ensure()
        obs = _encode_for(self._model, view)
        mask = action_mask(view, legal)
        idx, self._state = self._model.predict(
            obs,
            state=self._state,
            episode_start=np.array([self._episode_start]),
            deterministic=self.deterministic,
            action_masks=mask,
        )
        self._episode_start = False
        return index_to_action(view, legal, int(idx))

    def reset(self, seed=None) -> None:
        self._state = None
        self._episode_start = True
