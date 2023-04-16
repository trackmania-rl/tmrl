# === Trackmania =======================================================================================================


# standard library imports

# third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from math import floor, sqrt
from torch.nn import Conv2d, Module, ModuleList
# import torchvision

# local imports
from tmrl.util import prod
from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg


# SUPPORTED ============================================================================================================


# Spinup MLP: =======================================================
# Adapted from the SAC implementation of OpenAI Spinup


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
EPSILON = 1e-7


class SquashedGaussianMLPActor(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
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
            return a.cpu().numpy()


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
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.cpu().numpy()


# REDQ MLP: =====================================================


class REDQMLPActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 hidden_sizes=(256, 256),
                 activation=nn.ReLU,
                 n=10):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.n = n
        self.qs = ModuleList([MLPQFunction(observation_space, action_space, hidden_sizes, activation) for _ in range(self.n)])

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.cpu().numpy()


# CNNs: ================================================================================================================

# EfficientNet =========================================================================================================

# EfficientNetV2 implementation adapted from https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
# We use the EfficientNetV2 structure for image features and we merge the TM2020 float features to linear layers


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, nb_channels_in=3, dim_output=1, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(nb_channels_in, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, dim_output)

        self._initialize_weights()

    def forward(self, x):
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
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


class SquashedGaussianEffNetActor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        self.cnn = effnetv2_s(nb_channels_in=4, dim_output=247, width_mult=1.).float()
        self.net = mlp([256, 256], [nn.ReLU, nn.ReLU])
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        imgs_tensor = obs[3].float()
        float_tensors = (obs[0], obs[1], obs[2], *obs[4:])
        float_tensor = torch.cat(float_tensors, -1).float()
        cnn_out = self.cnn(imgs_tensor)
        mlp_in = torch.cat((cnn_out, float_tensor), -1)
        net_out = self.net(mlp_in)
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
        import sys
        size = sys.getsizeof(obs)
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.cpu().numpy()


class EffNetQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class EffNetActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.cpu().numpy()


# Vanilla CNN FOR GRAYSCALE IMAGES: ====================================================================================


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


class VanillaCNN(Module):
    def __init__(self, q_net):
        super(VanillaCNN, self).__init__()
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
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

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
        assert flat_features == self.flat_features, f"x.shape:{x.shape}, flat_features:{flat_features}, self.out_channels:{self.out_channels}, self.h_out:{self.h_out}, self.w_out:{self.w_out}"
        x = x.view(-1, flat_features)
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)
        return x


class SquashedGaussianVanillaCNNActor(TorchActorModule):
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
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # NB: this is from Spinup:
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)  # FIXME: this formula is mathematically wrong, no idea why it seems to work
            # Whereas SB3 does this:
            # logp_pi -= torch.sum(torch.log(1 - torch.tanh(pi_action) ** 2 + EPSILON), dim=1)  # TODO: double check
            # # log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.cpu().numpy()


class VanillaCNNQFunction(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)

    def forward(self, obs, act):
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class VanillaCNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # build policy and value functions
        self.actor = SquashedGaussianVanillaCNNActor(observation_space, action_space)
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.cpu().numpy()


# Vanilla CNN FOR COLOR IMAGES: ========================================================================================

def remove_colors(images):
    """
    We remove colors so that we can simply use the same structure as the grayscale model.

    The "color" default pipeline is mostly here for support, as our model effectively gets rid of 2 channels out of 3.
    If you actually want to use colors, do not use the default pipeline.
    Instead, you need to code a custom model that doesn't get rid of them.
    """
    images = images[:, :, :, :, 0]
    return images


class SquashedGaussianVanillaColorCNNActor(SquashedGaussianVanillaCNNActor):
    def forward(self, obs, test=False, with_logprob=True):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, test=False, with_logprob=True)


class VanillaColorCNNQFunction(VanillaCNNQFunction):
    def forward(self, obs, act):
        speed, gear, rpm, images, act1, act2 = obs
        images = remove_colors(images)
        obs = (speed, gear, rpm, images, act1, act2)
        return super().forward(obs, act)


class VanillaColorCNNActorCritic(VanillaCNNActorCritic):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # build policy and value functions
        self.actor = SquashedGaussianVanillaColorCNNActor(observation_space, action_space)
        self.q1 = VanillaColorCNNQFunction(observation_space, action_space)
        self.q2 = VanillaColorCNNQFunction(observation_space, action_space)


# UNSUPPORTED ==========================================================================================================


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
    def __init__(self, obs_space, act_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
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
            return a.cpu().numpy()


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
    def __init__(self, observation_space, action_space, rnn_size=100, rnn_len=2, mlp_sizes=(100, 100), activation=nn.ReLU):
        super().__init__()

        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianRNNActor(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q1 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
        self.q2 = RNNQFunction(observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation)
