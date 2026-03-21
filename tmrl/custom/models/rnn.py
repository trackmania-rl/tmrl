"""RNN-based actor-critic models.

This module contains actor-critic implementations using recurrent neural
networks for sequential decision making.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from tmrl.custom.models.base import LOG_STD_MAX, LOG_STD_MIN, mlp
from tmrl.util import prod


def rnn(input_size, rnn_size, rnn_len):
    """
    sizes is ignored for now, expect first values and length
    """
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_rnn_layers,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    )
    return gru


class SquashedGaussianRNNActor(nn.Module):
    """RNN-based actor with Gaussian policy and tanh squashing."""

    def __init__(
        self,
        obs_space,
        act_space,
        rnn_size=100,
        rnn_len=2,
        mlp_sizes=(100, 100),
        activation=nn.ReLU,
    ):
        super().__init__()

        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        act_limit = act_space.high[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size, *list(mlp_sizes)], activation, activation)
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.act_limit = act_limit
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, obs_seq, test=False, with_logprob=True, save_hidden=False):
        """
        obs: observation
        h: hidden state
        Returns:
            pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()

        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        obs_seq_cat = torch.cat(obs_seq, -1)
        net_out, h = self.rnn(obs_seq_cat, h)
        net_out = net_out[:, -1]
        net_out = self.mlp(net_out)
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

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h = h

        return pi_action, logp_pi

    def act(self, obs, test=False):
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)
        with torch.no_grad():
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.squeeze().cpu().numpy()


class RNNQFunction(nn.Module):
    """
    The action is merged in the latent space after the RNN
    """

    def __init__(
        self,
        obs_space,
        act_space,
        rnn_size=100,
        rnn_len=2,
        mlp_sizes=(100, 100),
        activation=nn.ReLU,
    ):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act, *list(mlp_sizes), 1], activation)
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len

    def forward(self, obs_seq, act, save_hidden=False):
        """
        obs: observation
        h: hidden state
        Returns:
            pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()

        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        obs_seq_cat = torch.cat(obs_seq, -1)

        net_out, h = self.rnn(obs_seq_cat, h)
        net_out = net_out[:, -1]
        net_out = torch.cat((net_out, act), -1)
        q = self.mlp(net_out)

        if save_hidden:
            self.h = h

        return torch.squeeze(q, -1)


class RNNActorCritic(nn.Module):
    """Actor-critic using RNN for sequential processing."""

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_size=100,
        rnn_len=2,
        mlp_sizes=(100, 100),
        activation=nn.ReLU,
    ):
        super().__init__()

        self.actor = SquashedGaussianRNNActor(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation
        )
        self.q1 = RNNQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation
        )
        self.q2 = RNNQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation
        )

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()
