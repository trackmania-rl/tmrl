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
# from tmrl.custom.custom_models import ActorModule, MlpActionValue, SacLinear, prod
import logging


# === Trackmania =======================================================================================================


# standard library imports
from dataclasses import InitVar, dataclass

# third-party imports
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn import Linear, Module, ModuleList, ReLU, Sequential

# local imports
from tmrl.nn import SacLinear, TanhNormalLayer
from tmrl.util import collate, partition, prod
from tmrl.actor import ActorModule
import logging


# Adapted from the SAC implementation of OpenAI Spinup


class MlpActionValue(Sequential):
    def __init__(self, obs_space, act_space, hidden_units):
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        super().__init__(SacLinear(dim_obs + dim_act, hidden_units), ReLU(), SacLinear(hidden_units, hidden_units), ReLU(), Linear(hidden_units, 2))

    # noinspection PyMethodOverriding
    def forward(self, obs, action):
        # logging.debug(f" obs:{obs}")
        x = torch.cat((*obs, action), -1)
        return super().forward(x)


class MlpPolicy(Sequential):
    def __init__(self, obs_space, act_space, hidden_units=256, act_buf_len=0):
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        super().__init__(SacLinear(dim_obs, hidden_units), ReLU(), SacLinear(hidden_units, hidden_units), ReLU(), TanhNormalLayer(hidden_units, dim_act))

    # noinspection PyMethodOverriding
    def forward(self, obs):
        # logging.debug(f" obs:{obs}")
        return super().forward(torch.cat(obs, -1))  # XXX


class Mlp(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256, num_critics: int = 2, act_buf_len=0):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple), f"{observation_space}"
        self.critics = ModuleList(MlpActionValue(observation_space, action_space, hidden_units) for _ in range(num_critics))
        self.actor = MlpPolicy(observation_space, action_space, hidden_units)
        self.critic_output_layers = [c[-1] for c in self.critics]


# Spinup MLP: =======================================================


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(ActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU, act_buf_len=0):
        super().__init__(observation_space, action_space)
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        net_out = self.net(torch.cat(obs, -1))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.numpy()


class MLPQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU, act_buf_len=0):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.numpy()


# REDQ MLP: =====================================================


class REDQMLPActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 act_buf_len=0,
                 n=10):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation, act_limit)
        self.n = n
        self.qs = ModuleList([MLPQFunction(observation_space, action_space, hidden_sizes, activation) for _ in range(self.n)])

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.numpy()

# RNN: ==========================================================


def rnn(input_size, rnn_size, rnn_len):
    """
    sizes is ignored for now, expect first values and length
    """
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_rnn_layers, bias=True, batch_first=True, dropout=0, bidirectional=False)
    return gru


class SquashedGaussianRNNActor(nn.Module):
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU, act_buf_len=0):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        act_limit = act_space.high[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size] + list(mlp_sizes), activation, activation)
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

        # sequence_len = obs_seq[0].shape[0]
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

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h = h

        return pi_action, logp_pi

    def act(self, obs, test=False):
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.numpy()


class RNNQFunction(nn.Module):
    """
    The action is merged in the latent space after the RNN
    """
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act] + list(mlp_sizes) + [1], activation)
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

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        # logging.debug(f"len(obs_seq):{len(obs_seq)}")
        # logging.debug(f"obs_seq[0].shape:{obs_seq[0].shape}")
        # logging.debug(f"obs_seq[1].shape:{obs_seq[1].shape}")
        # logging.debug(f"obs_seq[2].shape:{obs_seq[2].shape}")
        # logging.debug(f"obs_seq[3].shape:{obs_seq[3].shape}")

        obs_seq_cat = torch.cat(obs_seq, -1)

        # logging.debug(f"obs_seq_cat.shape:{obs_seq_cat.shape}")

        net_out, h = self.rnn(obs_seq_cat, h)
        # logging.debug(f"1 net_out.shape:{net_out.shape}")
        net_out = net_out[:, -1]
        # logging.debug(f"2 net_out.shape:{net_out.shape}")
        net_out = torch.cat((net_out, act), -1)
        # logging.debug(f"3 net_out.shape:{net_out.shape}")
        q = self.mlp(net_out)

        if save_hidden:
            self.h = h

        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class RNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU, act_buf_len=0):
        super().__init__()

        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianRNNActor(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation, act_limit)
        self.q1 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q2 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)


# CNN: ==========================================================


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
