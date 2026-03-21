from math import floor
from typing import cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal
from torch.nn import Conv2d, Module

import tmrl.config as cfg
from tmrl.actor import TorchActorModule
from tmrl.custom.models.model_constants import LOG_STD_MAX, LOG_STD_MIN, effnetv2_s, effnetv2_xs


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


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
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


if hasattr(nn, "SiLU"):
    SiLU = nn.SiLU
else:

    class SiLU(nn.Module):  # type: ignore[no-redef]
        @staticmethod
        def forward(x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), SiLU())


def conv_dw_3x3_bn(inp, oup, stride):
    """Depthwise 3x3 + pointwise 1x1 stem; fewer ops than full conv_3x3_bn."""
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        SiLU(),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU(),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), SiLU())


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ResidualMLPBlock(nn.Module):
    """Residual block: Linear->LN->Swish->Linear->LN->Swish; out = x + scale * block(x).

    Uses 1/sqrt(num_blocks) scaling to prevent activation accumulation in deep stacks,
    and zero-initializes the second linear so blocks start as near-identity (ReZero-like).
    """

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = SiLU()
        self.scale = scale
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.linear1.weight, gain=1.0)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.ln2(out)
        return cast(torch.Tensor, x + self.scale * self.act(out))


def residual_mlp_backbone(
    input_dim: int,
    hidden_dim: int,
    num_blocks: int,
) -> nn.Module:
    """Build residual MLP: input_proj -> num_blocks x ResidualMLPBlock. Output dim = hidden_dim.

    Each block's residual contribution is scaled by 1/sqrt(num_blocks) to prevent
    activation magnitude growth in deep stacks.
    """
    layers: list[nn.Module] = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.LayerNorm(hidden_dim))
    layers.append(SiLU())
    scale = 1.0 / max(1, num_blocks) ** 0.5
    for _ in range(num_blocks):
        layers.append(ResidualMLPBlock(hidden_dim, scale=scale))
    return nn.Sequential(*layers)


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Project ``x`` onto the unit hypersphere along ``dim``."""
    result: torch.Tensor = x / (x.norm(dim=dim, keepdim=True) + eps)
    return result


class HypersphericalLinear(nn.Module):
    """Linear layer with unit-norm weight rows and a learnable scaling vector.

    After each optimizer step, call `project_weights()` to re-normalize rows
    to the unit hypersphere (or use the SimbaV2Backbone wrapper that does it).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scaler = nn.Parameter(torch.ones(out_features))
        nn.init.orthogonal_(self.weight)
        self.project_weights()

    @torch.no_grad()
    def project_weights(self):
        self.weight.copy_(_l2_normalize(self.weight, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight) * self.scaler


class SimbaV2Block(nn.Module):
    """SimbaV2 residual block: inverted-bottleneck MLP + LERP + L2-norm.

    Architecture (per paper Eq.11-12):
      h_tilde = l2_norm(W2 * relu(W1 * h * s))    -- inverted bottleneck
      h_out   = l2_norm((1 - alpha) * h + alpha * h_tilde)  -- LERP
    """

    def __init__(self, dim: int, expand_ratio: int = 4):
        super().__init__()
        inner_dim = dim * expand_ratio
        self.up_proj = HypersphericalLinear(dim, inner_dim)
        self.down_proj = HypersphericalLinear(inner_dim, dim)
        self.alpha = nn.Parameter(torch.full((dim,), 0.5))

    def project_weights(self):
        self.up_proj.project_weights()
        self.down_proj.project_weights()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_tilde = self.up_proj(h)
        h_tilde = torch.relu(h_tilde)
        h_tilde = self.down_proj(h_tilde)
        h_tilde = _l2_normalize(h_tilde)
        alpha = self.alpha.sigmoid()
        out = (1.0 - alpha) * h + alpha * h_tilde
        return _l2_normalize(out)


class SimbaV2Backbone(nn.Module):
    """SimbaV2 backbone: shift + L2-norm input, hyperspherical residual blocks.

    Replaces residual_mlp_backbone when USE_SIMBAV2=true in config.
    Call `project_weights()` after each optimizer step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int, c_shift: float = 1.0):
        super().__init__()
        self.c_shift = c_shift
        self.input_proj = HypersphericalLinear(input_dim + 1, hidden_dim)
        self.blocks = nn.ModuleList([SimbaV2Block(hidden_dim) for _ in range(num_blocks)])

    @torch.no_grad()
    def project_weights(self):
        self.input_proj.project_weights()
        for block in self.blocks:
            block.project_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.full((*x.shape[:-1], 1), self.c_shift, device=x.device, dtype=x.dtype)
        x = torch.cat([x, shift], dim=-1)
        x = _l2_normalize(x)
        x = self.input_proj(x)
        x = _l2_normalize(x)
        for block in self.blocks:
            x = block(x)
        return x


def simba_v2_backbone(
    input_dim: int,
    hidden_dim: int,
    num_blocks: int,
    c_shift: float = 1.0,
) -> SimbaV2Backbone:
    """Build SimbaV2 backbone. Output dim = hidden_dim (features on the unit hypersphere)."""
    return SimbaV2Backbone(input_dim, hidden_dim, num_blocks, c_shift=c_shift)


_EFFNET_VARIANTS = {
    "xs": effnetv2_xs,
    "s": effnetv2_s,
}


class FrozenEfficientNetEncoder(nn.Module):
    """
    EfficientNetV2 feature extractor. Can be frozen (no gradients) or trainable.

    - frozen=True (default): All parameters have requires_grad=False, forward uses
      torch.no_grad(). Use for a fixed feature extractor (e.g. USE_FROZEN_EFFNET=true).
      Encoder kept in train mode so BatchNorm uses batch statistics (running stats
      not calibrated when frozen from random init).
    - frozen=False: Encoder is trainable. Use config USE_FROZEN_EFFNET=false to train
      the image encoder end-to-end with the rest of the model.

    Supported variants: "xs" (8 blocks, fast), "s" (40 blocks, original).
    use_dw_stem=True: depthwise-separable first conv (faster). FROZEN_EFFNET_USE_DW_STEM.
    """

    def __init__(
        self,
        nb_channels_in: int = 4,
        embed_dim: int = 256,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
        frozen: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self._frozen = frozen
        factory = _EFFNET_VARIANTS.get(variant)
        if factory is None:
            raise ValueError(
                f"Unknown EfficientNet variant '{variant}', choose from {list(_EFFNET_VARIANTS)}"
            )
        self._encoder = factory(
            nb_channels_in=nb_channels_in,
            dim_output=embed_dim,
            width_mult=width_mult,
            use_dw_stem=use_dw_stem,
        )
        if frozen:
            for p in self._encoder.parameters():
                p.requires_grad = False
            # Keep in train mode: BatchNorm must use batch statistics since running
            # stats are never calibrated (encoder is frozen from random init).
            self._encoder.train()

    def train(self, mode: bool = True):
        """When frozen: always keep encoder in train mode for BatchNorm. When trainable: normal."""
        super().train(mode)
        if self._frozen:
            self._encoder.train()
        else:
            self._encoder.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, embed_dim) embeddings. When frozen, forward is no_grad."""
        if x.max() > 1.5:
            x = x / 255.0
        if self._frozen:
            with torch.no_grad():
                return cast(torch.Tensor, self._encoder(x))
        return cast(torch.Tensor, self._encoder(x))


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = (
        h_in
        + 2 * conv_layer.padding[0]
        - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1)
        - 1
    )
    h_out = floor(h_out / conv_layer.stride[0] + 1)
    w_out = (
        w_in
        + 2 * conv_layer.padding[1]
        - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1)
        - 1
    )
    w_out = floor(w_out / conv_layer.stride[1] + 1)
    return h_out, w_out


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


class VanillaCNN(Module):
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
            act = None

        x = functional.relu(self.conv1(images))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.conv4(x))
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, (
            f"x.shape:{x.shape}, flat_features:{flat_features}, self"
            f".out_channels:{self.out_channels}, self.h_out:{self.h_out}, "
            f"self.w_out:{self.w_out} "
        )
        x = x.view(-1, flat_features)
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)
        return x


class SquashedGaussianEffNetActor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        self.cnn = effnetv2_s(nb_channels_in=4, dim_output=247, width_mult=1.0).float()
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

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # Clamp pre-tanh to avoid -inf/NaN in squash log-prob correction
            pi_action_for_corr = pi_action.clamp(-20.0, 20.0)
            logp_pi -= (
                2 * (np.log(2) - pi_action_for_corr - functional.softplus(-2 * pi_action_for_corr))
            ).sum(axis=1)
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
