"""Residual MLP-based actor-critic models.

This module contains actor-critic implementations using residual MLP backbones
for improved gradient flow in deeper networks.
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
    _cat_obs,
    _obs_dim,
)
from tmrl.custom.models.model_blocks import residual_mlp_backbone

_LOG2 = float(np.log(2.0))
_SQUASH_CLAMP = 20.0


def _is_tuple_obs(obs_space) -> bool:
    """Check whether observation_space is a tuple of spaces."""
    try:
        sum(s for s in obs_space[0].shape for _ in obs_space)
        return True
    except TypeError:
        return False


def _act_from_forward(model: nn.Module, obs, test: bool = False) -> np.ndarray:
    """Shared deterministic act() for residual actor-critics."""
    with torch.no_grad():
        a, _ = model.forward(obs, test, False)
        res = a.squeeze().cpu().numpy()
        if not len(res.shape):
            res = np.expand_dims(res, 0)
        return np.asarray(res)


class SquashedGaussianResidualMLPActor(TorchActorModule):
    """Actor with residual MLP backbone (LayerNorm + Swish, 4-8 blocks)."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim=256,
        num_blocks=6,
    ):
        super().__init__(observation_space, action_space)
        self.tuple_obs = _is_tuple_obs(observation_space)
        dim_obs = _obs_dim(observation_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.backbone = residual_mlp_backbone(dim_obs, hidden_dim, num_blocks)
        self.mu_layer = nn.Linear(hidden_dim, dim_act)
        self.log_std_layer = nn.Linear(hidden_dim, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        x = _cat_obs(obs, self.tuple_obs)
        net_out = self.backbone(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            pi_action_for_corr = pi_action.clamp(-_SQUASH_CLAMP, _SQUASH_CLAMP)
            logp_pi -= (2 * (_LOG2 - pi_action_for_corr - F.softplus(-2 * pi_action_for_corr))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi

    def act(self, obs, test=False):
        return _act_from_forward(self, obs, test)


class ResidualMLPQFunction(nn.Module):
    """Q-function with residual MLP backbone."""

    def __init__(self, obs_space, act_space, hidden_dim=256, num_blocks=6):
        super().__init__()
        self.tuple_obs = _is_tuple_obs(obs_space)
        obs_dim = _obs_dim(obs_space)
        act_dim = act_space.shape[0]
        self.backbone = residual_mlp_backbone(obs_dim + act_dim, hidden_dim, num_blocks)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        x = (
            torch.cat((*obs, act), -1)
            if self.tuple_obs
            else torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        )
        out = self.backbone(x)
        q = self.q_head(out)
        return torch.squeeze(q, -1)


class ResidualMLPActorCritic(nn.Module):
    """Actor-critic with residual MLP (depth 4-8 blocks, width 256). For Lidar + SAC."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim=256,
        num_blocks=6,
    ):
        super().__init__()
        self.actor = SquashedGaussianResidualMLPActor(
            observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
        )
        self.q1 = ResidualMLPQFunction(
            observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
        )
        self.q2 = ResidualMLPQFunction(
            observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
        )

    def act(self, obs, test=False):
        return _act_from_forward(self.actor, obs, test)


class REDQResidualMLPActorCritic(nn.Module):
    """REDQ with residual MLP (for 2-actor sample efficiency)."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim=256,
        num_blocks=6,
        n=10,
    ):
        super().__init__()
        self.actor = SquashedGaussianResidualMLPActor(
            observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
        )
        self.n = n
        self.qs = ModuleList(
            [
                ResidualMLPQFunction(
                    observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
                )
                for _ in range(self.n)
            ]
        )

    def act(self, obs, test=False):
        return _act_from_forward(self.actor, obs, test)
