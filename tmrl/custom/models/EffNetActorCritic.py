from math import sqrt

import torch
from torch import nn

from tmrl.custom.models.MLPActorCritic import MLPQFunction, SquashedGaussianMLPActor
from tmrl.custom.models.model_blocks import (
    MBConv,
    _make_divisible,
    conv_1x1_bn,
    conv_3x3_bn,
    conv_dw_3x3_bn,
    mlp,
)
from tmrl.util import prod


class EffNetV2(nn.Module):
    """
    Description: Defines an EfficientNetV2-style convolutional neural network architecture.
    """

    def __init__(self, cfgs, nb_channels_in=3, dim_output=1, width_mult=1.0, use_dw_stem=False):
        """
        Description: Initializes an EfficientNetV2-style convolutional neural network.
        Arguments:
        cfgs: Configuration for building the layers.
        nb_channels_in: Number of input channels (default: 3 for RGB).
        dim_output: Dimension of the output (default: 1).
        width_mult: Width multiplier for controlling the model width (default: 1.0).
        use_dw_stem: If True, use depthwise-separable first conv (faster, slightly fewer params).
        """
        super().__init__()
        self.cfgs = cfgs

        input_channel = _make_divisible(24 * width_mult, 8)
        stem = (
            conv_dw_3x3_bn(nb_channels_in, input_channel, 2)
            if use_dw_stem
            else conv_3x3_bn(nb_channels_in, input_channel, 2)
        )
        layers = [stem]

        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, dim_output)

        self._initialize_weights()

    def forward(self, x):
        """
        Description: Defines the forward pass through the network layers.
        Arguments:
        x: Input tensor.
        Returns: Output tensor after passing through the network layers.
        """
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


class EffNetQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        """
        Description: Initializes the Q-function (critic) architecture for the Actor-Critic method.
        Arguments:
        obs_space: Observation space for the environment.
        act_space: Action space for the environment.
        hidden_sizes: Hidden layer sizes in the MLP (default: (256, 256)).
        activation: Activation function used in the MLP (default: nn.ReLU).
        """
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim, *list(hidden_sizes), 1], activation)

    def forward(self, obs, act):
        """
        Description: Defines the forward pass of the Q-function network.
        Arguments:
        obs: Observation tensor.
        act: Action tensor.
        Returns: Q-values tensor for the given observations and actions.
        """
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class EffNetActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU
    ):
        """
        Initializes Actor-Critic with EfficientNetV2-style actor and Q-function nets.
        Arguments:
        observation_space: Observation space for the environment.
        action_space: Action space for the environment.
        hidden_sizes: Hidden layer sizes in the networks (default: (256, 256)).
        activation: Activation function used in the networks (default: nn.ReLU).
        """
        super().__init__()

        self.actor = SquashedGaussianMLPActor(
            observation_space, action_space, hidden_sizes, activation
        )
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        """
        Description: Produces actions from the actor network based on observed states.
        Arguments:
        obs: Observation tensor.
        test: Flag for testing mode (default: False).
        Returns: Array of actions inferred from the actor network.
        """
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.cpu().numpy()
