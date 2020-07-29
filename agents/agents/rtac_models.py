from dataclasses import dataclass, InitVar

import gym
import torch
from agents.util import collate, partition
from agents.nn import TanhNormalLayer, SacLinear, big_conv
from torch.nn import Module, Linear, Sequential, ReLU, Conv2d, LeakyReLU
from torch.nn.functional import leaky_relu

from agents.sac_models import ActorModule


# TODO: Add separate mlp model


class Mlp(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        input_dim = sum(s.shape[0] for s in observation_space)
        self.net = Sequential(
            # SacLinear(input_dim, hidden_units), ReLU(),
            # SacLinear(hidden_units, hidden_units), ReLU(),
            Linear(input_dim, hidden_units), ReLU(),
            Linear(hidden_units, hidden_units), ReLU(),
        )
        self.critic_layer = Linear(hidden_units, 2)  # predict future reward and entropy separately
        self.actor_layer = TanhNormalLayer(hidden_units, action_space.shape[0])
        self.critic_output_layers = (self.critic_layer,)

    def actor(self, x):
        return self(x)[0]

    def forward(self, x):
        assert isinstance(x, tuple)
        x = torch.cat(x, dim=1)
        h = self.net(x)
        v = self.critic_layer(h)
        action_distribution = self.actor_layer(h)
        return action_distribution, (v,), (h,)


class DoubleActorModule(ActorModule):
    @property
    def critic_output_layers(self):
        return self.a.critic_output_layers + self.b.critic_output_layers

    def actor(self, x):
        return self.a(x)[0]

    def forward(self, x):
        action_distribution, v0, h0 = self.a(x)
        _, v1, h1 = self.b(x)
        return action_distribution, v0 + v1, h0 + h1  # note that the + here is not addition but tuple concatenation!


class MlpDouble(DoubleActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256):
        super().__init__()
        self.a = Mlp(observation_space, action_space, hidden_units=hidden_units)
        self.b = Mlp(observation_space, action_space, hidden_units=hidden_units)


class ConvDouble(DoubleActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 512, conv: type = big_conv):
        super().__init__()
        self.a = ConvRTAC(observation_space, action_space, hidden_units=hidden_units, Conv=conv)
        self.b = ConvRTAC(observation_space, action_space, hidden_units=hidden_units, Conv=conv)


class ConvRTAC(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 512, Conv: type = big_conv):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        (img_sp, vec_sp), ac_sp = observation_space

        self.conv = Conv(img_sp.shape[0])

        with torch.no_grad():
            conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)

        self.lin1 = Linear(conv_size + vec_sp.shape[0] + ac_sp.shape[0], hidden_units)
        self.lin2 = Linear(hidden_units + vec_sp.shape[0] + ac_sp.shape[0], hidden_units)
        self.critic_layer = Linear(hidden_units, 2)  # predict future reward and entropy separately
        self.actor_layer = TanhNormalLayer(hidden_units, action_space.shape[0])
        self.critic_output_layers = (self.critic_layer,)

    def forward(self, inp):
        (x, vec), action = inp
        x = x.type(torch.float32)
        x = x / 255 - 0.5
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = leaky_relu(self.lin1(torch.cat((x, vec, action), -1)))
        x = leaky_relu(self.lin2(torch.cat((x, vec, action), -1)))
        v = self.critic_layer(x)
        action_distribution = self.actor_layer(x)
        return action_distribution, (v,), (x,)


class ConvSeparate(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 512, conv: type = big_conv):
        super().__init__()
        self.a = ConvRTAC(observation_space, action_space, hidden_units=hidden_units, Conv=conv)
        self.b = ConvRTAC(observation_space, action_space, hidden_units=hidden_units, Conv=conv)
        self.c = ConvRTAC(observation_space, action_space, hidden_units=hidden_units, Conv=conv)

    @property
    def critic_output_layers(self):
        return self.b.critic_output_layers + self.c.critic_output_layers

    def actor(self, x):
        return self.a(x)[0]

    def forward(self, x):
        action_distribution, *_ = self.a(x)
        _, v0, h0 = self.b(x)
        _, v1, h1 = self.c(x)
        return action_distribution, v0 + v1, h0 + h1  # note that the + here is not addition but tuple concatenation!
