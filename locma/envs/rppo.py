"""MaskableRecurrentPPO — the maskable + recurrent hybrid sb3-contrib doesn't ship.

sb3-contrib provides MaskablePPO (action masking, no memory) and RecurrentPPO
(LSTM memory, no masking) as separate algorithms. LOCMA's 155-slot semantic
action space is mostly illegal at any decision point, so masking is not
optional — an unmasked recurrent policy would spend its probability mass on
illegal actions. This module grafts the maskable distribution and the
action-mask buffer plumbing onto the recurrent classes:

  - ``MaskableRecurrentActorCriticPolicy`` — RecurrentActorCriticPolicy with a
    MaskableDistribution head; every path that builds a distribution accepts
    ``action_masks``. The logits layer is identical to the unmasked one, so
    the swap changes no parameters.
  - ``MaskableRecurrent(Dict)RolloutBuffer`` — the recurrent buffers plus a
    per-step action-mask store. Padded timesteps get all-ones masks so the
    masked categorical never sees an all-illegal row (which would produce
    NaNs; padded entries are excluded from every loss term via the sequence
    mask anyway).
  - ``MaskableRecurrentPPO`` — RecurrentPPO with masks fetched from the env
    (``action_masks()`` method, as MaskablePPO's ``get_action_masks`` does)
    during collection and replayed through ``evaluate_actions`` in training.

``collect_rollouts`` and ``train`` are copies of RecurrentPPO's (sb3-contrib
2.9.0) with the mask plumbing added — there is no narrower override seam.

Only Discrete action spaces are supported (all LOCMA needs).
"""

from __future__ import annotations

from copy import deepcopy
from typing import NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import (
    MaskableDistribution,
    make_masked_proba_distribution,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.common.recurrent.buffers import (
    RecurrentDictRolloutBuffer,
    RecurrentRolloutBuffer,
)
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import explained_variance, obs_as_tensor


class MaskableRecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor
    action_masks: th.Tensor


class MaskableRecurrentDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor
    action_masks: th.Tensor


class _MaskStoreMixin:
    """Shared action-mask store for the two recurrent buffer subclasses.

    Relies on the host buffer's ``action_space``, ``buffer_size``, ``n_envs``,
    ``pos``, ``generator_ready``, ``swap_and_flatten`` and (post-`_get_samples`)
    ``pad`` attributes.
    """

    def _reset_masks(self) -> None:
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("MaskableRecurrentPPO supports Discrete action spaces only")
        self.mask_dims = int(self.action_space.n)
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32
        )

    def _store_masks(self, action_masks: np.ndarray | None) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))

    def _flatten_masks_once(self) -> None:
        # Piggy-back on the parent's generator_ready flag: flatten BEFORE the
        # parent's get() flips it, exactly once per rollout.
        if not self.generator_ready:
            self.action_masks = self.swap_and_flatten(self.action_masks)

    def _padded_masks(self, batch_inds: np.ndarray, padded_batch_size: int) -> th.Tensor:
        # Pad with 1.0 (all-legal): an all-zero mask row would drive every
        # logit to -inf and NaN the categorical. Padded rows are excluded from
        # all losses by the sequence mask, so their content is otherwise inert.
        return self.pad(self.action_masks[batch_inds], padding_value=1.0).reshape(
            padded_batch_size, self.mask_dims
        )


class MaskableRecurrentRolloutBuffer(RecurrentRolloutBuffer, _MaskStoreMixin):
    """RecurrentRolloutBuffer that also stores per-step action masks."""

    def reset(self) -> None:
        super().reset()
        self._reset_masks()

    def add(self, *args, action_masks: np.ndarray | None = None, **kwargs) -> None:
        self._store_masks(action_masks)
        super().add(*args, **kwargs)

    def get(self, batch_size: int | None = None):
        self._flatten_masks_once()
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env_change, env=None):
        base = super()._get_samples(batch_inds, env_change, env)
        masks = self._padded_masks(batch_inds, base.actions.shape[0])
        return MaskableRecurrentRolloutBufferSamples(*base, masks)


class MaskableRecurrentDictRolloutBuffer(RecurrentDictRolloutBuffer, _MaskStoreMixin):
    """RecurrentDictRolloutBuffer that also stores per-step action masks."""

    def reset(self) -> None:
        super().reset()
        self._reset_masks()

    def add(self, *args, action_masks: np.ndarray | None = None, **kwargs) -> None:
        self._store_masks(action_masks)
        super().add(*args, **kwargs)

    def get(self, batch_size: int | None = None):
        self._flatten_masks_once()
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env_change, env=None):
        base = super()._get_samples(batch_inds, env_change, env)
        masks = self._padded_masks(batch_inds, base.actions.shape[0])
        return MaskableRecurrentDictRolloutBufferSamples(*base, masks)


class MaskableRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """RecurrentActorCriticPolicy with a maskable action distribution.

    The parent builds ``action_net`` for a plain CategoricalDistribution; for
    Discrete spaces the maskable net is the identical ``nn.Linear``, so
    swapping ``action_dist`` post-init changes no parameters and the optimizer
    (already constructed) stays valid.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dist = make_masked_proba_distribution(self.action_space)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        # Copy of RecurrentActorCriticPolicy.forward + apply_masking.
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        action_masks: np.ndarray | None = None,
    ) -> tuple[MaskableDistribution, tuple[th.Tensor, ...]]:
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi, lstm_states = self._process_sequence(
            features, lstm_states, episode_starts, self.lstm_actor
        )
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution, lstm_states

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Copy of RecurrentActorCriticPolicy.evaluate_actions + apply_masking.
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...]]:
        distribution, lstm_states = self.get_distribution(
            observation, lstm_states, episode_starts, action_masks
        )
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        # Copy of RecurrentActorCriticPolicy.predict + action_masks pass-through.
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        if state is None:
            state = np.concatenate(
                [np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1
            )
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            states = (
                th.tensor(state[0], dtype=th.float32, device=self.device),
                th.tensor(state[1], dtype=th.float32, device=self.device),
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states


class MaskableRecurrentMultiInputPolicy(MaskableRecurrentActorCriticPolicy):
    """Dict-obs variant (CombinedExtractor default, swap in TokenSetExtractor
    via ``policy_kwargs["features_extractor_class"]``)."""

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        kwargs.setdefault("features_extractor_class", CombinedExtractor)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)


class MaskableRecurrentPPO(RecurrentPPO):
    """RecurrentPPO + action masking (see module docstring).

    The training env must expose an ``action_masks()`` method (BattleEnv does),
    per sb3-contrib's MaskablePPO convention.
    """

    policy_aliases = {
        "MlpLstmPolicy": MaskableRecurrentActorCriticPolicy,
        "MultiInputLstmPolicy": MaskableRecurrentMultiInputPolicy,
    }

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.policy, MaskableRecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass MaskableRecurrentActorCriticPolicy")
        # Replace the parent's buffer with the mask-carrying variant (same ctor
        # geometry; the parent built it two lines earlier so this is cheap).
        lstm = self.policy.lstm_actor
        buffer_cls = (
            MaskableRecurrentDictRolloutBuffer
            if isinstance(self.observation_space, spaces.Dict)
            else MaskableRecurrentRolloutBuffer
        )
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size),
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        # Copy of RecurrentPPO.collect_rollouts (2.9.0) + action-mask plumbing.
        assert isinstance(
            rollout_buffer, (MaskableRecurrentRolloutBuffer, MaskableRecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent+maskable policies"
        assert self._last_obs is not None, "No previous observation was provided"
        if not is_masking_supported(env):
            raise ValueError("The env does not expose action masks (action_masks() method)")

        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            action_masks = get_action_masks(env)
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(
                    self._last_episode_starts, dtype=th.float32, device=self.device
                )
                actions, values, log_probs, lstm_states = self.policy(
                    obs_tensor, lstm_states, episode_starts, action_masks=action_masks
                )

            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with the value function
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        ep_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_lstm_state, ep_starts
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                action_masks=action_masks,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        # Copy of RecurrentPPO.train (2.9.0); the single change is passing
        # rollout_data.action_masks into evaluate_actions.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (
                        advantages[mask].std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} "
                            f"due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ):
        return self.policy.predict(
            observation, state, episode_start, deterministic, action_masks=action_masks
        )
