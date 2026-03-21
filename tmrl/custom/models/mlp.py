"""MLP-based actor-critic models.

This module contains standard multi-layer perceptron implementations for
SAC-style actor-critic architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn import ModuleList

from tmrl.actor import TorchActorModule
from tmrl.custom.models.base import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    mlp,
)
from tmrl.util import prod


class SquashedGaussianMLPActor(TorchActorModule):
    """Gaussian policy with tanh squashing; logprob uses SAC appendix C correction."""

    def __init__(
        self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU
    ):
        super().__init__(observation_space, action_space)
        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs, *list(hidden_sizes)], activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            pi_action_for_corr = pi_action.clamp(-20.0, 20.0)
            logp_pi -= (
                2 * (np.log(2) - pi_action_for_corr - F.softplus(-2 * pi_action_for_corr))
            ).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class MLPQFunction(nn.Module):
    """Q-function using standard MLP architecture."""

    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        try:
            obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(obs_space.shape)
            self.tuple_obs = False
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim, *list(hidden_sizes), 1], activation)

    def forward(self, obs, act):
        x = (
            torch.cat((*obs, act), -1)
            if self.tuple_obs
            else torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        )
        q = self.q(x)
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):
    """MLP actor-critic (actor + two Q-networks)."""

    def __init__(
        self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU
    ):
        super().__init__()

        self.actor = SquashedGaussianMLPActor(
            observation_space, action_space, hidden_sizes, activation
        )
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class REDQMLPActorCritic(nn.Module):
    """REDQ agent: one actor, n Q-networks."""

    def __init__(
        self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU, n=10
    ):
        super().__init__()

        self.actor = SquashedGaussianMLPActor(
            observation_space, action_space, hidden_sizes, activation
        )
        self.n = n
        self.qs = ModuleList(
            [
                MLPQFunction(observation_space, action_space, hidden_sizes, activation)
                for _ in range(self.n)
            ]
        )

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()
