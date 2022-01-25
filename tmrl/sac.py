# standard library imports
import time
from copy import copy, deepcopy
from dataclasses import InitVar, dataclass
from functools import lru_cache, reduce

# third-party imports
import numpy as np
import torch
from torch.nn.functional import mse_loss

# local imports
import tmrl.sac_models
from tmrl.nn import PopArt, copy_shared, exponential_moving_average, hd_conv, no_grad
from tmrl.util import cached_property, collate, partial
import logging


@dataclass(eq=0)
class SacAgent:  # SAC agent with PopArt
    Env: InitVar

    Model: type = tmrl.sac_models.Mlp
    OutputNorm: type = PopArt
    lr_actor: float = 0.0003  # learning rate
    lr_critic: float = 0.0003  # learning rate
    discount: float = 0.99  # reward discount factor
    target_update: float = 0.005  # parameter for exponential moving average
    reward_scale: float = 5.
    entropy_scale: float = 1.
    device: str = None

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.Model(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.model.critics.parameters(), lr=self.lr_critic)

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

    def train(self, batch):

        obs, actions, rewards, next_obs, dones = batch
        new_action_distribution = self.model.actor(obs)  # outputs distribution object
        new_actions = new_action_distribution.rsample()  # samples using the reparametrization trick

        # critic loss
        next_action_distribution = self.model_nograd.actor(next_obs)  # outputs distribution object
        next_actions = next_action_distribution.sample()  # samples
        next_value = [c(next_obs, next_actions) for c in self.model_target.critics]
        next_value = reduce(torch.min, next_value)  # minimum action-value
        next_value = self.outputnorm_target.unnormalize(next_value)  # PopArt (not present in the original paper)
        # next_value = self.outputnorm.unnormalize(next_value)  # PopArt (not present in the original paper)

        # predict entropy rewards in a separate dimension from the normal rewards (not present in the original paper)
        next_action_entropy = -(1. - dones) * self.discount * next_action_distribution.log_prob(next_actions)
        reward_components = torch.cat((
            self.reward_scale * rewards[:, None],
            self.entropy_scale * next_action_entropy[:, None],
        ), dim=1)  # shape = (batch_size, reward_components)

        value_target = reward_components + (1. - dones[:, None]) * self.discount * next_value
        normalized_value_target = self.outputnorm.update(value_target)  # PopArt update and normalize

        values = [c(obs, actions) for c in self.model.critics]
        assert values[0].shape == normalized_value_target.shape and not normalized_value_target.requires_grad
        loss_critic = sum(mse_loss(v, normalized_value_target) for v in values)

        # update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # actor loss
        new_value = [c(obs, new_actions) for c in self.model.critics]  # new_actions with reparametrization trick
        new_value = reduce(torch.min, new_value)  # minimum action_values
        # assert new_value.shape == (self.batch_size, 2)
        assert new_value.shape[1] == 2

        new_value = self.outputnorm.unnormalize(new_value)
        new_value[:, -1] -= self.entropy_scale * new_action_distribution.log_prob(new_actions)
        loss_actor = -self.outputnorm.normalize_sum(new_value.sum(1)).mean()  # normalize_sum preserves relative scale

        # update actor
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # update target critics and normalizers
        exponential_moving_average(self.model_target.critics.parameters(), self.model.critics.parameters(), self.target_update)
        exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)

        return dict(
            loss_actor=loss_actor.detach(),
            loss_critic=loss_critic.detach(),
            outputnorm_reward_mean=self.outputnorm.mean[0],
            outputnorm_entropy_mean=self.outputnorm.mean[-1],
            outputnorm_reward_std=self.outputnorm.std[0],
            outputnorm_entropy_std=self.outputnorm.std[-1],
        )
