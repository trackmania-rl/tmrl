"""Base utilities for neural network models.

This module contains common helper functions and constants used across
all model implementations.
"""

from math import floor, sqrt

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d

from tmrl.util import prod


def combined_shape(length, shape=None):
    """Return shape tuple combining length with an optional inner shape."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    """Create a multi-layer perceptron with specified sizes and activations."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    """Count total number of parameters in a module."""
    return sum([np.prod(p.shape) for p in module.parameters()])


# Constants for log standard deviation bounds in Gaussian policies
LOG_STD_MAX = 2
LOG_STD_MIN = -20  # Expanded support for high-variance exploration
EPSILON = 1e-7


def num_flat_features(x):
    """Calculate number of flat features in a tensor (excluding batch dim)."""
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer: Conv2d, h_in: int, w_in: int) -> tuple[int, int]:
    """Calculate output dimensions of a Conv2d layer given input dimensions."""
    # Extract values with explicit int conversion for mypy
    pad_h = (
        int(conv_layer.padding[0])
        if isinstance(conv_layer.padding, tuple)
        else int(conv_layer.padding)
    )
    pad_w = (
        int(conv_layer.padding[1])
        if isinstance(conv_layer.padding, tuple)
        else int(conv_layer.padding)
    )
    dil_h = (
        int(conv_layer.dilation[0])
        if isinstance(conv_layer.dilation, tuple)
        else int(conv_layer.dilation)
    )
    dil_w = (
        int(conv_layer.dilation[1])
        if isinstance(conv_layer.dilation, tuple)
        else int(conv_layer.dilation)
    )
    stride_h = (
        int(conv_layer.stride[0])
        if isinstance(conv_layer.stride, tuple)
        else int(conv_layer.stride)
    )
    stride_w = (
        int(conv_layer.stride[1])
        if isinstance(conv_layer.stride, tuple)
        else int(conv_layer.stride)
    )
    kernel_h = (
        int(conv_layer.kernel_size[0])
        if isinstance(conv_layer.kernel_size, tuple)
        else int(conv_layer.kernel_size)
    )
    kernel_w = (
        int(conv_layer.kernel_size[1])
        if isinstance(conv_layer.kernel_size, tuple)
        else int(conv_layer.kernel_size)
    )

    h_out = floor((h_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1)
    w_out = floor((w_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1)
    return h_out, w_out


def _obs_dim(observation_space):
    """Calculate total observation dimension from observation space."""
    try:
        return sum(prod(s for s in space.shape) for space in observation_space)
    except TypeError:
        return prod(observation_space.shape)


def _cat_obs(obs, tuple_obs):
    """Concatenate observations, handling both tuple and single tensor cases."""
    if tuple_obs:
        return torch.cat(obs, -1)
    return torch.flatten(obs, start_dim=1)


def _ensure_float(t: torch.Tensor) -> torch.Tensor:
    """Cast to float32 only when dtype is not already float (avoids redundant copies)."""
    if t.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return t
    return t.float()


def _obs_spaces_list(observation_space):
    """Convert observation space to a list of spaces."""
    if hasattr(observation_space, "spaces"):
        return list(observation_space.spaces)
    return list(observation_space)


def _vector_dim_except(observation_space, image_index: int):
    """Sum of flattened sizes of all observation components except the one at image_index."""
    try:
        spaces = list(observation_space)
    except TypeError:
        return prod(observation_space.shape)
    return sum(
        prod(space.shape) if hasattr(space, "shape") else int(space)
        for i, space in enumerate(spaces)
        if i != image_index
    )


def _cat_obs_except_image(obs, image_index: int):
    """Concatenate all observation tensors except obs[image_index] on the last dim."""
    parts = [obs[i] for i in range(len(obs)) if i != image_index]
    if any(t.dim() > 2 for t in parts):
        parts = [t.view(t.size(0), -1) for t in parts]
    return _ensure_float(torch.cat(parts, dim=-1))


# EfficientNet utility functions


def _make_divisible(v, divisor, min_value=None):
    """Channel count divisible by divisor; round down at most 10% (tf slim mobilenet)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    """3x3 convolution with batch normalization and SiLU activation."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )


def conv_1x1_bn(inp, oup):
    """1x1 convolution with batch normalization and SiLU activation."""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU(),
    )


# SiLU activation (with fallback for older PyTorch versions)
if hasattr(nn, "SiLU"):
    SiLU = nn.SiLU
else:

    class SiLU(nn.Module):  # type: ignore[no-redef]
        """SiLU activation (Swish-1): x * sigmoid(x)."""

        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for EfficientNet."""

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


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution (MBConv) block for EfficientNet."""

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


class EffNetV2(nn.Module):
    """EfficientNetV2 architecture."""

    def __init__(self, cfgs, nb_channels_in=3, dim_output=1, width_mult=1.0):
        super().__init__()
        self.cfgs = cfgs

        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(nb_channels_in, input_channel, 2)]
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


def effnetv2_xs(**kwargs):
    """
    Constructs an EfficientNetV2-XS (extra-small) model.
    8 total blocks vs 40 in S — ~5x faster forward pass.
    """
    cfgs = [
        [1, 16, 1, 1, 0],
        [4, 32, 2, 2, 0],
        [4, 48, 2, 2, 0],
        [4, 96, 3, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
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
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)
