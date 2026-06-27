"""MaskablePPO-backed battle policy with lazy model loading (requires [ml])."""

from __future__ import annotations

from locma.envs.encode import action_mask, encode_battle, index_to_action


def _encoder(obs_mode: str):
    if obs_mode == "base":
        return lambda view, legal: encode_battle(view)
    if obs_mode == "tactical":
        from locma.envs.encode_tactical import encode_battle as encode_tactical  # noqa: PLC0415

        return encode_tactical
    raise ValueError(f"unknown obs_mode {obs_mode!r}")


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
        obs_mode: str = "base",
    ):
        self.model_path = model_path
        self.name = name
        self.deterministic = deterministic
        self.obs_mode = obs_mode
        self._encode_battle = _encoder(obs_mode)
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

            self._model = MaskablePPO.load(self.model_path)

    def battle_action(self, view, legal, state=None):
        self._ensure()
        obs = self._encode_battle(view, legal)
        mask = action_mask(view, legal)
        idx, _ = self._model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return index_to_action(view, legal, int(idx))

    def reset(self, seed=None) -> None:
        pass
