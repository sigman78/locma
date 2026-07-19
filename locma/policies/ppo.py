"""MaskablePPO-backed battle policy with lazy model loading (requires [ml])."""

from __future__ import annotations

from locma.envs.encode import (
    action_mask,
    draft_action_mask,
    encode_battle,
    encode_battle_tokens,
    encode_draft,
    index_to_action,
)


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


class MaskablePPOEnsembleBattlePolicy:
    """Mean-of-policy-heads ensemble over several MaskablePPO battle nets (E26).

    E8 found mean-of-critics ensembling the single biggest zero-training gain
    on the ``vbeam`` planner rung; this is the POLICY-head analog for the bare
    reactive net: batch each member's masked action distribution for the
    current view (same trunk-forward recipe as ``vbeam.NetValueEvaluator.
    _forward`` — ``obs_to_tensor``, ``extract_features``, ``mlp_extractor``,
    ``_get_action_dist_from_latent``, ``apply_masking``,
    ``distribution.probs``), mean the probabilities across members, and take
    the argmax. Deterministic by construction (no sampling). Members are
    loaded lazily on first use, exactly like ``MaskablePPOBattlePolicy``, so
    construction never touches the filesystem or imports the ``[ml]`` stack.
    """

    def __init__(self, model_paths: list[str], name: str = "ppo-ens"):
        if len(model_paths) < 2:
            raise ValueError("MaskablePPOEnsembleBattlePolicy needs at least 2 model paths")
        self.model_paths = list(model_paths)
        self.name = name
        self._models: list | None = None

    def _ensure(self) -> None:
        if self._models is None:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

            self._models = [MaskablePPO.load(p) for p in self.model_paths]

    def battle_action(self, view, legal, state=None):
        import numpy as np  # noqa: PLC0415 — lazy [ml]-adjacent dep
        import torch  # noqa: PLC0415 — lazy [ml] dep

        self._ensure()
        mask = action_mask(view, legal)
        probs = []
        for model in self._models:
            obs = _encode_for(model, view)
            policy = model.policy
            if isinstance(obs, dict):
                batch = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            else:
                batch = obs[None]
            obs_t, _ = policy.obs_to_tensor(batch)
            with torch.no_grad():
                features = policy.extract_features(obs_t)
                latent_pi, _latent_vf = policy.mlp_extractor(features)
                dist = policy._get_action_dist_from_latent(latent_pi)
                dist.apply_masking(mask[None])
                probs.append(dist.distribution.probs.cpu().numpy()[0])
        mean_probs = np.mean(np.stack(probs), axis=0)
        idx = int(np.argmax(mean_probs))
        return index_to_action(view, legal, idx)

    def reset(self, seed=None) -> None:
        pass


class MaskablePPODraftPolicy:
    """Wraps a saved MaskablePPO DRAFT model (train_draft, E18b) as a draft policy.

    Stateful like BalancedDraftPolicy: tracks its own picks (the deck-so-far
    summary is part of the draft observation), cleared on reset. Lazy model
    load, deterministic by default — same conventions as the battle wrapper.
    """

    def __init__(
        self,
        model_path: str = "draft.zip",
        name: str = "ppo-draft",
        deterministic: bool = True,
        model=None,
    ):
        self.model_path = model_path
        self.name = name
        self.deterministic = deterministic
        self._model = model
        self._picks: list = []

    def _ensure(self) -> None:
        if self._model is None:
            from sb3_contrib import MaskablePPO  # noqa: PLC0415 — optional [ml] dep

            self._model = MaskablePPO.load(self.model_path)

    def note_pick(self, view, idx) -> None:
        """Record a pick made on this policy's behalf (PartialRandomDraftPolicy
        hook), keeping the deck-so-far observation accurate."""
        self._picks.append(view.offered[idx])

    def draft_action(self, view, legal):
        self._ensure()
        obs = encode_draft(view, self._picks)
        mask = draft_action_mask(legal)
        idx, _ = self._model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        idx = int(idx)
        self.note_pick(view, idx)
        return idx

    def reset(self, seed=None) -> None:
        self._picks = []
