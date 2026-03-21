"""Vanilla CNN-based actor-critic models.

This module contains actor-critic implementations using standard CNN
architectures for image processing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn import Conv2d

import tmrl.config as cfg
from tmrl.actor import TorchActorModule
from tmrl.custom.models.base import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    conv2d_out_dims,
    mlp,
    num_flat_features,
)


class VanillaCNN(nn.Module):
    """Vanilla CNN feature extractor for TrackMania observations."""

    def __init__(self, q_net):
        super().__init__()
        self.q_net = q_net
        self.h_out, self.w_out = cfg.IMG_HEIGHT, cfg.IMG_WIDTH
        hist = cfg.IMG_HIST_LEN

        self.conv1 = Conv2d(hist, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out
        self.mlp_input_features = self.flat_features + 12 if self.q_net else self.flat_features + 9
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features, *self.mlp_layers], nn.ReLU)

    def forward(self, x):
        if self.q_net:
            speed, gear, rpm, images, act1, act2, act = x
        else:
            speed, gear, rpm, images, act1, act2 = x

        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, (
            f"x.shape:{x.shape}, flat_features:{flat_features}, "
            f"out_channels:{self.out_channels}, h_out:{self.h_out}, w_out:{self.w_out}"
        )
        x = x.view(-1, flat_features)
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)
        return x


class SquashedGaussianVanillaCNNActor(TorchActorModule):
    """Actor using Vanilla CNN for image processing."""

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = VanillaCNN(q_net=False)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        net_out = self.net(obs)
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
            return a.squeeze().cpu().numpy()


class VanillaCNNQFunction(nn.Module):
    """Q-function using Vanilla CNN for image processing."""

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)

    def forward(self, obs, act):
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)


class VanillaCNNActorCritic(nn.Module):
    """Actor-critic using Vanilla CNN backbone."""

    def __init__(self, observation_space, action_space):
        super().__init__()

        self.actor = SquashedGaussianVanillaCNNActor(observation_space, action_space)
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()


def remove_colors(images):
    """
    We remove colors so that we can simply use the same structure as the grayscale model.

    The "color" default pipeline is mostly here for support;
    our model effectively gets rid of 2 channels out of 3.
    If you actually want to use colors, do not use the default pipeline.
    Instead, you need to code a custom model that doesn't get rid of them.
    """
    images = images[:, :, :, :, 0]
    return images


class SquashedGaussianVanillaColorCNNActor(SquashedGaussianVanillaCNNActor):
    """Actor using Vanilla CNN with color image support (converts to grayscale)."""

    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, test=False, with_logprob=True)


class VanillaColorCNNQFunction(VanillaCNNQFunction):
    """Q-function using Vanilla CNN with color image support (converts to grayscale)."""

    def forward(self, obs, act):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, act)


class VanillaColorCNNActorCritic(VanillaCNNActorCritic):
    """Actor-critic using Vanilla CNN with color image support."""

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        self.actor = SquashedGaussianVanillaColorCNNActor(observation_space, action_space)
        self.q1 = VanillaColorCNNQFunction(observation_space, action_space)
        self.q2 = VanillaColorCNNQFunction(observation_space, action_space)
