from __future__ import annotations

from locma.envs.encode import action_mask, encode_battle, index_to_action
from locma.policies.random_policy import RandomPolicy


class SB3Policy:
    """MaskablePPO-backed policy wrapper with lazy model loading.

    Parameters
    ----------
    model_path:
        Path to a saved MaskablePPO ``.zip`` file.  The model is NOT loaded
        until the first call to :meth:`battle_action`, so construction works
        even when the file does not yet exist.
    name:
        Policy name; defaults to ``"sb3"``.
    draft:
        Draft-phase policy to delegate to; defaults to
        ``RandomPolicy("sb3-draft")``.
    """

    def __init__(self, model_path: str, name: str | None = None, draft=None) -> None:
        self.model_path = model_path
        self.name = name or "sb3"
        self.draft = draft or RandomPolicy("sb3-draft")
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure(self) -> None:
        """Lazy-load the MaskablePPO model on first use."""
        if self._model is None:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415  (lazy import)

            self._model = MaskablePPO.load(self.model_path)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def draft_action(self, view, legal):
        """Delegate draft decisions to the draft policy."""
        return self.draft.draft_action(view, legal)

    def battle_action(self, view, legal):
        """Encode observation, run masked predict, return the chosen action."""
        self._ensure()
        obs = encode_battle(view)
        mask = action_mask(legal)
        idx, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        return index_to_action(int(idx), legal)

    def reset(self, seed=None) -> None:
        """Reset the draft policy's RNG state."""
        self.draft.reset(seed)
