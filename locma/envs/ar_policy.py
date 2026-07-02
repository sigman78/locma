"""MaskablePPO policy with a conditional autoregressive action head.

Keeps the action space Discrete(155) and the 155-bool mask; swaps only the
single action head for three conditional heads. See
docs/ppo-autoreg-action-design.md. Requires the [ml] extra."""

from __future__ import annotations

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from locma.envs.ar_distribution import ARHeads, ar_evaluate, ar_sample


class MaskableAutoregressivePolicy(MaskableActorCriticPolicy):
    """Flat-obs MaskablePPO policy whose action head is autoregressive."""

    def _build(self, lr_schedule) -> None:
        # Build mlp_extractor, value_net, (unused) action_net, and optimizer.
        super()._build(lr_schedule)
        latent_dim = self.mlp_extractor.latent_dim_pi
        self.ar_heads = ARHeads(latent_dim)
        # Re-create the optimizer so it owns the AR head parameters too.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _masks_tensor(self, action_masks, batch: int) -> torch.Tensor:
        m = torch.as_tensor(action_masks, device=self.device)
        return m.reshape(batch, -1).bool()

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf

    def forward(self, obs, deterministic=False, action_masks=None):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        masks = self._masks_tensor(action_masks, latent_pi.shape[0])
        actions, log_prob = ar_sample(self.ar_heads, latent_pi, masks, deterministic)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        masks = self._masks_tensor(action_masks, latent_pi.shape[0])
        log_prob, entropy = ar_evaluate(self.ar_heads, latent_pi, masks, actions.long().reshape(-1))
        return values, log_prob, entropy

    def _predict(self, observation, deterministic=False, action_masks=None):
        latent_pi, _ = self._latents(observation)
        masks = self._masks_tensor(action_masks, latent_pi.shape[0])
        actions, _ = ar_sample(self.ar_heads, latent_pi, masks, deterministic)
        return actions
