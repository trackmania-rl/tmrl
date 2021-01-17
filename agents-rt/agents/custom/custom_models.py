# from dataclasses import InitVar, dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from agents.nn import TanhNormalLayer
from torch.nn import Linear, Sequential, ReLU, ModuleList, Module, Conv2d, MaxPool2d
import gym
from agents.sac_models import ActorModule
from agents.sac_models import prod, SacLinear, MlpActionValue


# === Trackmania =======================================================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, (8, 8))
        self.conv2 = nn.Conv2d(8, 16, (4, 4))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3, 3))
        self.fc1 = nn.Linear(672, 253)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (4, 4))
        x = F.max_pool2d(F.relu(self.conv2(x)), (4, 4))
        x = F.max_pool2d(F.relu(self.conv3(x)), (4, 4))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class TMModuleResnet(Module):
    def __init__(self, observation_space, action_space, is_q_network, act_buf_len=0):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        torch.autograd.set_detect_anomaly(True)
        self.img_dims = observation_space[3].shape
        self.vel_dim = observation_space[0].shape[0]
        self.gear_dim = observation_space[1].shape[0]
        self.rpm_dim = observation_space[2].shape[0]
        self.is_q_network = is_q_network
        self.act_buf_len = act_buf_len
        self.act_dim = action_space.shape[0]

        #self.cnn = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
        #self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 253)  # remove the last fc layer

        #self.cnn = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)
        #self.cnn.classifier[1] = nn.Linear(self.cnn.classifier[1].in_features, 253)

        self.cnn = Net()

        dim_fc1 = 253 + self.vel_dim + self.gear_dim + self.rpm_dim
        if self.is_q_network:
            dim_fc1 += self.act_dim
        if self.act_buf_len:
            dim_fc1 += self.act_dim * self.act_buf_len
        self.fc1 = Linear(dim_fc1, 256)

    def forward(self, x):
        assert isinstance(x, tuple), f"x is not a tuple: {x}"
        vel = x[0].float()
        gear = x[1].float()
        rpm = x[2].float()
        im1 = x[3].float()[:, 0]
        im2 = x[3].float()[:, 1]
        im3 = x[3].float()[:, 2]
        im4 = x[3].float()[:, 3]
        if self.act_buf_len:
            prev_act1 = x[4].float()  # TODO: change this for any act_buf_len
            prev_act2 = x[5].float()
        else:
            raise NotImplementedError
        im = torch.cat((im1, im2, im3, im4), dim=2)  # TODO : check device
        im = self.cnn(im)
        if self.is_q_network:
            act = x[-1].float()
            h = torch.cat((im, vel, gear, rpm, prev_act1, prev_act2, act), dim=1)
        else:
            h = torch.cat((im, vel, gear, rpm, prev_act1, prev_act2), dim=1)
        h = self.fc1(h)
        return h


class TMActionValue(Sequential):
    def __init__(self, observation_space, action_space, act_buf_len=0):
        super().__init__(
            TMModuleResnet(observation_space, action_space, is_q_network=True, act_buf_len=act_buf_len), ReLU(),
            Linear(256, 256), ReLU(),
            Linear(256, 2)  # we separate reward components
        )

    # noinspection PyMethodOverriding
    def forward(self, obs, action):
        x = (*obs, action)
        res = super().forward(x)
        # print(f"DEBUG: av res:{res}")
        return res


class TMPolicy(Sequential):
    def __init__(self, observation_space, action_space, act_buf_len=0):
        super().__init__(
            TMModuleResnet(observation_space, action_space, is_q_network=False, act_buf_len=act_buf_len), ReLU(),
            Linear(256, 256), ReLU(),
            TanhNormalLayer(256, action_space.shape[0])
        )

    # noinspection PyMethodOverriding
    def forward(self, obs):
        # res = super().forward(torch.cat(obs, 1))
        res = super().forward(obs)
        # print(f"DEBUG: po res:{res}")
        return res


class Tm_hybrid_1(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256, num_critics: int = 2, act_buf_len=0):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple), f"{observation_space} is not a spaces.Tuple"
        self.critics = ModuleList(TMActionValue(observation_space, action_space, act_buf_len=act_buf_len) for _ in range(num_critics))
        self.actor = TMPolicy(observation_space, action_space, act_buf_len=act_buf_len)
        self.critic_output_layers = [c[-1] for c in self.critics]
