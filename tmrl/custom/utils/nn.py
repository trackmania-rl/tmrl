# standard library imports
from copy import deepcopy
from dataclasses import InitVar, dataclass

# third-party imports
import numpy as np
import torch
from torch.distributions import Distribution, Normal
from torch.nn import Module
from torch.nn.init import calculate_gain, kaiming_uniform_, xavier_uniform_
from torch.nn.parameter import Parameter

# local imports
from tmrl.util import partial


def detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    else:
        return [detach(elem) for elem in x]


def no_grad(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def exponential_moving_average(averages, values, factor):
    with torch.no_grad():
        for a, v in zip(averages, values):
            a += factor * (v - a)  # equivalent to a = (1-factor) * a + factor * v


def copy_shared(model_a):
    """Create a deepcopy of a model but with the underlying state_dict shared. E.g. useful in combination with `no_grad`."""
    model_b = deepcopy(model_a)
    sda = model_a.state_dict(keep_vars=True)
    sdb = model_b.state_dict(keep_vars=True)
    for key in sda:
        a, b = sda[key], sdb[key]
        b.data = a.data  # strangely this will not make a.data and b.data the same object but their underlying data_ptr will be the same
        assert b.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()
    return model_b


class PopArt(Module):
    """PopArt http://papers.nips.cc/paper/6076-learning-values-across-many-orders-of-magnitude"""
    def __init__(self, output_layer, beta: float = 0.0003, zero_debias: bool = True, start_pop: int = 8):
        # zero_debias=True and start_pop=8 seem to improve things a little but (False, 0) works as well
        super().__init__()
        self.start_pop = start_pop
        self.beta = beta
        self.zero_debias = zero_debias
        self.output_layers = output_layer if isinstance(output_layer, (tuple, list, torch.nn.ModuleList)) else (output_layer, )
        shape = self.output_layers[0].bias.shape
        device = self.output_layers[0].bias.device
        assert all(shape == x.bias.shape for x in self.output_layers)
        self.mean = Parameter(torch.zeros(shape, device=device), requires_grad=False)
        self.mean_square = Parameter(torch.ones(shape, device=device), requires_grad=False)
        self.std = Parameter(torch.ones(shape, device=device), requires_grad=False)
        self.updates = 0

    @torch.no_grad()
    def update(self, targets):
        beta = max(1 / (self.updates + 1), self.beta) if self.zero_debias else self.beta
        # note that for beta = 1/self.updates the resulting mean, std would be the true mean and std over all past data

        new_mean = (1 - beta) * self.mean + beta * targets.mean(0)
        new_mean_square = (1 - beta) * self.mean_square + beta * (targets * targets).mean(0)
        new_std = (new_mean_square - new_mean * new_mean).sqrt().clamp(0.0001, 1e6)

        # assert self.std.shape == (1,), 'this has only been tested in 1D'

        if self.updates >= self.start_pop:
            for layer in self.output_layers:
                layer.weight *= (self.std / new_std)[:, None]
                layer.bias *= self.std
                layer.bias += self.mean - new_mean
                layer.bias /= new_std

        self.mean.copy_(new_mean)
        self.mean_square.copy_(new_mean_square)
        self.std.copy_(new_std)
        self.updates += 1
        return self.normalize(targets)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean

    def normalize_sum(self, s):
        """normalize x.sum(1) preserving relative weightings between elements"""
        return (s - self.mean.sum()) / self.std.norm()


# noinspection PyAbstractClass
class TanhNormal(Distribution):
    """Distribution of X ~ tanh(Z) where Z ~ N(mean, std)
    Adapted from https://github.com/vitchyr/rlkit
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        super().__init__(self.normal.batch_shape, self.normal.event_shape)

    def log_prob(self, x):
        if hasattr(x, "pre_tanh_value"):
            pre_tanh_value = x.pre_tanh_value
        else:
            pre_tanh_value = (torch.log(1 + x + self.epsilon) - torch.log(1 - x + self.epsilon)) / 2
        assert x.dim() == 2 and pre_tanh_value.dim() == 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - x * x + self.epsilon)

    def sample(self, sample_shape=torch.Size()):
        z = self.normal.sample(sample_shape)
        out = torch.tanh(z)
        out.pre_tanh_value = z
        return out

    def rsample(self, sample_shape=torch.Size()):
        z = self.normal.rsample(sample_shape)
        out = torch.tanh(z)
        out.pre_tanh_value = z
        return out


# noinspection PyAbstractClass
class Independent(torch.distributions.Independent):
    def sample_test(self):
        return torch.tanh(self.base_dist.normal_mean)


class TanhNormalLayer(torch.nn.Module):
    def __init__(self, n, m):
        super().__init__()

        self.lin_mean = torch.nn.Linear(n, m)
        # self.lin_mean.weight.data
        # self.lin_mean.bias.data

        self.lin_std = torch.nn.Linear(n, m)
        self.lin_std.weight.data.uniform_(-1e-3, 1e-3)
        self.lin_std.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        mean = self.lin_mean(x)
        log_std = self.lin_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        # a = TanhTransformedDist(Independent(Normal(m, std), 1))
        a = Independent(TanhNormal(mean, std), 1)
        return a


class RlkitLinear(torch.nn.Linear):
    def __init__(self, *args):
        super().__init__(*args)
        # TODO: investigate the following
        # this mistake seems to be in rlkit too
        # https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
        fan_in = self.weight.shape[0]  # this is actually fanout!!!
        bound = 1. / np.sqrt(fan_in)
        self.weight.data.uniform_(-bound, bound)
        self.bias.data.fill_(0.1)


class SacLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        with torch.no_grad():
            self.weight.uniform_(-0.06, 0.06)  # 0.06 == 1 / sqrt(256)
            self.bias.fill_(0.1)


class BasicReLU(torch.nn.Linear):
    def forward(self, x):
        x = super().forward(x)
        return torch.relu(x)


class AffineReLU(BasicReLU):
    def __init__(self, in_features, out_features, init_weight_bound: float = 1., init_bias: float = 0.):
        super().__init__(in_features, out_features)
        bound = init_weight_bound / np.sqrt(in_features)
        self.weight.data.uniform_(-bound, bound)
        self.bias.data.fill_(init_bias)


class NormalizedReLU(torch.nn.Sequential):
    def __init__(self, in_features, out_features, prenorm_bias=True):
        super().__init__(torch.nn.Linear(in_features, out_features, bias=prenorm_bias), torch.nn.LayerNorm(out_features), torch.nn.ReLU())


class KaimingReLU(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        with torch.no_grad():
            kaiming_uniform_(self.weight)
            self.bias.fill_(0.)

    def forward(self, x):
        x = super().forward(x)
        return torch.relu(x)


Linear10 = partial(AffineReLU, init_bias=1.)
Linear04 = partial(AffineReLU, init_bias=0.4)
LinearConstBias = partial(AffineReLU, init_bias=0.1)
LinearZeroBias = partial(AffineReLU, init_bias=0.)
AffineSimon = partial(AffineReLU, init_weight_bound=0.01, init_bias=1.)


def dqn_conv(n):
    return torch.nn.Sequential(torch.nn.Conv2d(n, 32, kernel_size=8, stride=4), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, kernel_size=4, stride=2), torch.nn.ReLU(),
                               torch.nn.Conv2d(64, 64, kernel_size=3, stride=1), torch.nn.ReLU())


def big_conv(n):
    # if input shape = 64 x 256 then output shape = 2 x 26
    return torch.nn.Sequential(
        torch.nn.Conv2d(n, 64, 8, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 64, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 128, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(128, 128, 4, stride=1),
        torch.nn.LeakyReLU(),
    )


def hd_conv(n):
    return torch.nn.Sequential(
        torch.nn.Conv2d(n, 32, 8, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(32, 64, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 64, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 128, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(128, 128, 4, stride=2),
        torch.nn.LeakyReLU(),
    )
