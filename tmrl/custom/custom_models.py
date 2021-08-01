# from dataclasses import InitVar, dataclass
# standard library imports
from math import floor

# third-party imports
import gym
import torch
from torch.nn import Conv2d, Linear, MaxPool2d, Module, ModuleList, ReLU, Sequential
from torch.nn import functional as F

# local imports
from tmrl.nn import TanhNormalLayer
from tmrl.sac_models import ActorModule, MlpActionValue, SacLinear, prod
import logging
# === Trackmania =======================================================================================================


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(3, 8, (8, 8))
        self.conv2 = Conv2d(8, 16, (4, 4))
        self.conv3 = Conv2d(16, 32, (3, 3))
        self.conv4 = Conv2d(32, 64, (3, 3))
        self.fc1 = Linear(672, 253)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (4, 4))
        x = F.max_pool2d(F.relu(self.conv2(x)), (4, 4))
        x = F.max_pool2d(F.relu(self.conv3(x)), (4, 4))
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        return x


class DeepmindCNN(Module):
    def __init__(self, h_in, w_in, channels_in):
        super(DeepmindCNN, self).__init__()
        self.h_out, self.w_out = h_in, w_in

        self.conv1 = Conv2d(in_channels=channels_in, out_channels=32, kernel_size=(8, 8), stride=4, padding=0, dilation=1, bias=True, padding_mode='zeros')
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=0, dilation=1, bias=True, padding_mode='zeros')
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros')
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.out_channels = self.conv3.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out

        logging.debug(f" h_in:{h_in}, w_in:{w_in}, h_out:{self.h_out}, w_out:{self.w_out}, flat_features:{self.flat_features}")

    def forward(self, x):
        logging.debug(f" forward, shape x :{x.shape}")
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"
        x = x.view(-1, flat_features)
        return x


class BigCNN(Module):
    def __init__(self, h_in, w_in, channels_in):
        super(BigCNN, self).__init__()
        self.h_out, self.w_out = h_in, w_in

        self.conv1 = Conv2d(channels_in, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out

        logging.debug(f" h_in:{h_in}, w_in:{w_in}, h_out:{self.h_out}, w_out:{self.w_out}, flat_features:{self.flat_features}")

    def forward(self, x):  # TODO: Simon uses leaky relu instead of relu, see what works best
        # logging.debug(f" forward, shape x :{x.shape}")
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"
        x = x.view(-1, flat_features)
        return x


class TM20CNNModule(Module):
    def __init__(self, observation_space, action_space, is_q_network, act_buf_len=0):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        # torch.autograd.set_detect_anomaly(True)  # FIXME: remove for optimization
        self.img_dims = observation_space[3].shape
        self.vel_dim = observation_space[0].shape[0]
        self.gear_dim = observation_space[1].shape[0]
        self.rpm_dim = observation_space[2].shape[0]
        self.is_q_network = is_q_network
        self.act_buf_len = act_buf_len
        self.act_dim = action_space.shape[0]

        logging.debug(f" self.img_dims: {self.img_dims}")
        h_in = self.img_dims[2]
        w_in = self.img_dims[3]
        channels_in = self.img_dims[0] * self.img_dims[1]  # successive images as channels

        self.cnn = BigCNN(h_in=h_in, w_in=w_in, channels_in=channels_in)

        dim_fc1_in = self.cnn.flat_features + self.vel_dim + self.gear_dim + self.rpm_dim
        if self.is_q_network:
            dim_fc1_in += self.act_dim
        if self.act_buf_len:
            dim_fc1_in += self.act_dim * self.act_buf_len
        self.fc1 = Linear(dim_fc1_in, 512)

    def forward(self, x):
        # assert isinstance(x, tuple), f"x is not a tuple: {x}"
        vel = x[0].float()
        gear = x[1].float()
        rpm = x[2].float()
        ims = x[3].float()
        im1 = ims[:, 0]
        im2 = ims[:, 1]
        im3 = ims[:, 2]
        im4 = ims[:, 3]
        # logging.debug(f" forward: im1.shape:{im1.shape}")
        if self.act_buf_len:
            all_acts = torch.cat((x[4:]), dim=1).float()  # if q network, the last action will be act
        else:
            raise NotImplementedError
        cat_im = torch.cat((im1, im2, im3, im4), dim=1)  # cat on channel dimension  # TODO : check device
        h = self.cnn(cat_im)
        h = torch.cat((h, vel, gear, rpm, all_acts), dim=1)
        h = self.fc1(h)  # No ReLU here because this is done in the Sequential
        return h


class TMActionValue(Sequential):
    def __init__(self, observation_space, action_space, act_buf_len=0):
        super().__init__(
            TM20CNNModule(observation_space, action_space, is_q_network=True, act_buf_len=act_buf_len),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 2)  # we separate reward components
        )

    # noinspection PyMethodOverriding
    def forward(self, obs, action):
        x = (*obs, action)
        res = super().forward(x)
        # logging.debug(f" av res:{res}")
        return res


class TMPolicy(Sequential):
    def __init__(self, observation_space, action_space, act_buf_len=0):
        super().__init__(TM20CNNModule(observation_space, action_space, is_q_network=False, act_buf_len=act_buf_len), ReLU(), Linear(512, 256), ReLU(), TanhNormalLayer(256, action_space.shape[0]))

    # noinspection PyMethodOverriding
    def forward(self, obs):
        # res = super().forward(torch.cat(obs, 1))
        res = super().forward(obs)
        # logging.debug(f" po res:{res}")
        return res


class Tm_hybrid_1(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 512, num_critics: int = 2, act_buf_len=0):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple), f"{observation_space} is not a spaces.Tuple"
        self.critics = ModuleList(TMActionValue(observation_space, action_space, act_buf_len=act_buf_len) for _ in range(num_critics))
        self.actor = TMPolicy(observation_space, action_space, act_buf_len=act_buf_len)
        self.critic_output_layers = [c[-1] for c in self.critics]
