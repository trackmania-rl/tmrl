# from collections import deque
from copy import deepcopy, copy
from dataclasses import dataclass, InitVar
from functools import lru_cache, reduce
# from itertools import chain
import numpy as np
import torch
from torch.nn.functional import mse_loss

from tmrl.memory_dataloading import MemoryDataloading
from tmrl.nn import PopArt, no_grad, copy_shared, exponential_moving_average, hd_conv
from tmrl.util import cached_property, partial
import tmrl.sac_models

# import gym


@dataclass(eq=0)
class Agent:
    Env: InitVar

    Model: type = tmrl.sac_models.Mlp
    Memory: type = MemoryDataloading
    OutputNorm: type = PopArt
    batchsize: int = 256  # training batch size
    memory_size: int = 1000000  # replay memory size
    lr: float = 0.0003  # learning rate
    discount: float = 0.99  # reward discount factor
    target_update: float = 0.005  # parameter for exponential moving average
    reward_scale: float = 5.
    entropy_scale: float = 1.
    device: str = None

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    # total_updates = 0
    # environment_steps = 0

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.Model(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critics.parameters(), lr=self.lr)
        self.memory = self.Memory(self.memory_size, self.batchsize, device)

        # self.actor_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.actor_optimizer,self.lr/10,self.lr*10, step_size_up=2000)
        # self.critic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.critic_optimizer,self.lr / 10,self.lr * 10, step_size_up=2000)

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

    def act(self, state, obs, r, done, info, train=False):
        state = self.model.reset() if state is None else state  # initialize state if necessary
        action, next_state, _ = self.model.act(state, obs, r, done, info, train)
        if train:
            self.memory.append(np.float32(r), np.float32(done), info, obs, action)
        return action, next_state

    def train(self):
        obs, actions, rewards, next_obs, terminals = self.memory.sample()  # sample a transition from the replay buffer
        # print("DEBUG: sampling new action")
        new_action_distribution = self.model.actor(obs)  # outputs distribution object
        new_actions = new_action_distribution.rsample()  # samples using the reparametrization trick

        # critic loss
        # print("DEBUG: sampling next action")
        next_action_distribution = self.model_nograd.actor(next_obs)  # outputs distribution object
        next_actions = next_action_distribution.sample()  # samples
        # print("DEBUG: sampling next values")
        next_value = [c(next_obs, next_actions) for c in self.model_target.critics]
        next_value = reduce(torch.min, next_value)  # minimum action-value
        next_value = self.outputnorm_target.unnormalize(next_value)  # PopArt (not present in the original paper)
        # next_value = self.outputnorm.unnormalize(next_value)  # PopArt (not present in the original paper)

        # predict entropy rewards in a separate dimension from the normal rewards (not present in the original paper)
        next_action_entropy = - (1. - terminals) * self.discount * next_action_distribution.log_prob(next_actions)
        reward_components = torch.cat((
            self.reward_scale * rewards[:, None],
            self.entropy_scale * next_action_entropy[:, None],
        ), dim=1)  # shape = (batchsize, reward_components)

        value_target = reward_components + (1. - terminals[:, None]) * self.discount * next_value
        normalized_value_target = self.outputnorm.update(value_target)  # PopArt update and normalize

        # print("DEBUG: sampling old values")
        values = [c(obs, actions) for c in self.model.critics]
        assert values[0].shape == normalized_value_target.shape and not normalized_value_target.requires_grad
        loss_critic = sum(mse_loss(v, normalized_value_target) for v in values)

        # update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # actor loss
        # print("DEBUG: sampling new values")
        new_value = [c(obs, new_actions) for c in self.model.critics]  # new_actions with reparametrization trick
        new_value = reduce(torch.min, new_value)  # minimum action_values
        assert new_value.shape == (self.batchsize, 2)

        new_value = self.outputnorm.unnormalize(new_value)
        new_value[:, -1] -= self.entropy_scale * new_action_distribution.log_prob(new_actions)
        loss_actor = - self.outputnorm.normalize_sum(new_value.sum(1)).mean()  # normalize_sum preserves relative scale

        # update actor
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # update target critics and normalizers
        exponential_moving_average(self.model_target.critics.parameters(), self.model.critics.parameters(), self.target_update)
        exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)
        # self.actor_lr_scheduler.step()
        # self.critic_lr_scheduler.step()
        return dict(
            loss_actor=loss_actor.detach(),
            loss_critic=loss_critic.detach(),
            outputnorm_reward_mean=self.outputnorm.mean[0],
            outputnorm_entropy_mean=self.outputnorm.mean[-1],
            outputnorm_reward_std=self.outputnorm.std[0],
            outputnorm_entropy_std=self.outputnorm.std[-1],
            memory_size=len(self.memory),
        )
