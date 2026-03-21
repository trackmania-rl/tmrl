"""EfficientNet-based actor-critic models.

This module contains actor-critic implementations using frozen EfficientNet
backbones for image feature extraction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import tmrl.config as cfg
from tmrl.actor import TorchActorModule
from tmrl.custom.models.base import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    _cat_obs_except_image,
    _ensure_float,
    _obs_spaces_list,
    _vector_dim_except,
    effnetv2_s,
    mlp,
)
from tmrl.custom.models.model_blocks import FrozenEfficientNetEncoder, residual_mlp_backbone
from tmrl.util import prod


def _gnn_effnet_image_index(observation_space) -> int:
    """Find the image observation index in a GNN-EffNet style observation space."""
    spaces = _obs_spaces_list(observation_space)
    h, w = getattr(cfg, "IMG_HEIGHT", 64), getattr(cfg, "IMG_WIDTH", 64)
    for i, sp in enumerate(spaces):
        if hasattr(sp, "shape") and sp.shape is not None and len(sp.shape) >= 3:
            s = sp.shape
            if s[-2] == h and s[-1] == w:
                return i
    return len(spaces) - 1


def _gnn_effnet_physics_dims(observation_space, image_index: int):
    """Extract track and physics dimensions from GNN-EffNet observation space."""
    spaces = _obs_spaces_list(observation_space)
    dim_track = int(prod(spaces[0].shape))
    dim_physics = sum(int(prod(s.shape)) for s in spaces[1:image_index])
    return dim_track, dim_physics


class SquashedGaussianFrozenEffNetResidualActor(TorchActorModule):
    """Actor: frozen EfficientNet (image->embed) + concat vector -> residual MLP -> policy."""

    def __init__(
        self,
        observation_space,
        action_space,
        image_index: int = 3,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
    ):
        super().__init__(observation_space, action_space)
        self.image_index = image_index
        try:
            spaces = list(observation_space)
            nb_channels_in = int(prod(spaces[image_index].shape))
            if len(spaces[image_index].shape) >= 2:
                nb_channels_in = spaces[image_index].shape[0]
        except (TypeError, IndexError):
            nb_channels_in = 4
        vector_dim = _vector_dim_except(observation_space, image_index)
        input_dim = embed_dim + vector_dim
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        self.encoder = FrozenEfficientNetEncoder(
            nb_channels_in=nb_channels_in,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
        )
        self.backbone = residual_mlp_backbone(input_dim, hidden_dim, num_blocks)
        self.mu_layer = nn.Linear(hidden_dim, dim_act)
        self.log_std_layer = nn.Linear(hidden_dim, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        imgs = _ensure_float(obs[self.image_index])
        vec = _cat_obs_except_image(obs, self.image_index)
        emb = self.encoder(imgs)
        x = torch.cat([emb, vec], dim=-1)
        net_out = self.backbone(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        # Policy head in float32 for numerical stability
        mu, log_std = mu.float(), log_std.float()
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
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class FrozenEffNetResidualQFunction(nn.Module):
    """Q-function: frozen EffNet(image) + concat(vector, act) -> residual MLP -> Q."""

    def __init__(
        self,
        obs_space,
        act_space,
        image_index: int = 3,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
    ):
        super().__init__()
        self.image_index = image_index
        try:
            spaces = list(obs_space)
            nb_channels_in = (
                int(spaces[image_index].shape[0])
                if len(spaces[image_index].shape) >= 2
                else int(prod(spaces[image_index].shape))
            )
        except (TypeError, IndexError):
            nb_channels_in = 4
        vector_dim = _vector_dim_except(obs_space, image_index)
        act_dim = act_space.shape[0]
        input_dim = embed_dim + vector_dim + act_dim

        self.encoder = FrozenEfficientNetEncoder(
            nb_channels_in=nb_channels_in,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
        )
        self.backbone = residual_mlp_backbone(input_dim, hidden_dim, num_blocks)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        imgs = _ensure_float(obs[self.image_index])
        vec = _cat_obs_except_image(obs, self.image_index)
        emb = self.encoder(imgs)
        x = torch.cat([emb, vec, act], dim=-1)
        q = self.q_head(self.backbone(x))
        return torch.squeeze(q, -1)


class FrozenEffNetResidualActorCritic(nn.Module):
    """Actor-critic: frozen small EfficientNet embeddings + residual MLP head (SiLU, LayerNorm)."""

    def __init__(
        self,
        observation_space,
        action_space,
        image_index: int = 3,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
    ):
        super().__init__()
        self.actor = SquashedGaussianFrozenEffNetResidualActor(
            observation_space,
            action_space,
            image_index=image_index,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
        )
        self.q1 = FrozenEffNetResidualQFunction(
            observation_space,
            action_space,
            image_index=image_index,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
        )
        self.q2 = FrozenEffNetResidualQFunction(
            observation_space,
            action_space,
            image_index=image_index,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
        )

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class SquashedGaussianEffNetActor(TorchActorModule):
    """Actor using trainable EfficientNet-S as CNN backbone."""

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
        imgs_tensor = _ensure_float(obs[3])
        float_tensors = (obs[0], obs[1], obs[2], *obs[4:])
        float_tensor = _ensure_float(torch.cat(float_tensors, -1))
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


class EffNetQFunction(nn.Module):
    """Q-function using EfficientNet-S CNN backbone."""

    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim, *list(hidden_sizes), 1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)


class EffNetActorCritic(nn.Module):
    """Actor-critic with EfficientNet-S CNN backbone + MLP heads."""

    def __init__(
        self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU
    ):
        super().__init__()

        from tmrl.custom.models.mlp import MLPQFunction, SquashedGaussianMLPActor

        self.actor = SquashedGaussianMLPActor(
            observation_space, action_space, hidden_sizes, activation
        )
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()
