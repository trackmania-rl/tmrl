"""GNN + EfficientNet + Sophy hybrid actor-critic models.

This module contains actor-critic implementations combining GNN track encoders,
EfficientNet image encoders, and Sophy-style residual MLP backbones.
"""

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchrl.modules import NoisyLinear

import tmrl.config as cfg
import tmrl.config.config_objects as cfo
from tmrl.actor import TorchActorModule
from tmrl.custom.models.base import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    _ensure_float,
    _obs_spaces_list,
)
from tmrl.custom.models.efficientnet import _gnn_effnet_image_index, _gnn_effnet_physics_dims
from tmrl.custom.models.model_blocks import FrozenEfficientNetEncoder, residual_mlp_backbone
from tmrl.custom.utils.nn import GSDEModule
from tmrl.util import prod

_LOG2 = float(np.log(2.0))


class _TrackGNN(nn.Module):
    """Graph Neural Network for processing track point sequences."""

    def __init__(self, num_nodes: int, in_dim: int = 3, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        edge_src = torch.cat([torch.arange(num_nodes - 1), torch.arange(1, num_nodes)], dim=0)
        edge_dst = torch.cat([torch.arange(1, num_nodes), torch.arange(num_nodes - 1)], dim=0)
        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)
        self.node_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n = x.shape
        x = x.permute(0, 2, 1)
        h = self.node_in(x)
        edge_src = cast(torch.Tensor, self.edge_src)
        edge_dst = cast(torch.Tensor, self.edge_dst)
        for lin, norm in zip(self.layers, self.norms, strict=True):
            msg = h[:, edge_src]
            agg = torch.zeros(b, n, self.hidden_dim, device=h.device, dtype=h.dtype)
            agg.index_add_(1, edge_dst, msg)
            deg = torch.zeros(n, device=h.device, dtype=h.dtype)
            deg.index_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=h.dtype))
            deg = deg.clamp(min=1).view(1, -1, 1)
            agg = agg / deg
            h = norm(lin(torch.cat([h, agg], dim=-1)).relu())
        out = self.readout(h)
        return cast(torch.Tensor, out.mean(dim=1))


def _build_track_gnn_branch(dim_track: int, hidden_dim: int) -> nn.Module:
    """Build a GNN-based track encoding branch."""
    assert dim_track >= 3, "track dim must be at least 3"
    assert dim_track % 3 == 0, "track dim must be 6*N (3 channels)"
    num_nodes = dim_track // 3
    gnn_hidden = getattr(cfg, "GNN_HIDDEN", 64)
    gnn_layers = getattr(cfg, "GNN_LAYERS", 3)
    gnn = _TrackGNN(
        num_nodes=num_nodes,
        in_dim=3,
        hidden_dim=gnn_hidden,
        num_layers=gnn_layers,
    )
    return nn.Sequential(
        gnn,
        nn.Linear(gnn_hidden, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
    )


def _obs_to_flat_tensor(observation, batch_size: int) -> torch.Tensor:
    """
    Combines a list of observation tensors into a single flat tensor.

    Args:
        observation: Observation list or tuple of tensors.
        batch_size (int): Current batch size.

    Returns:
        torch.Tensor: Flattened and concatenated observation tensor.
    """
    if isinstance(observation, torch.Tensor):
        return observation.view(batch_size, -1).float()
    obs_list = list(observation)
    for i in range(len(obs_list)):
        obs_list[i] = obs_list[i].view(batch_size, -1)
    return torch.cat(obs_list, -1).float()


def _ensure_image_4d(imgs: torch.Tensor, image_index: int) -> None:
    """Validate that the image tensor has the expected 4D shape (N,C,H,W)."""
    if imgs.dim() == 2 and imgs.shape[-1] == 3:
        raise RuntimeError(
            f"Obs at index {image_index} has shape {tuple(imgs.shape)} (action, not image). "
            "Env may append action buffer after image. Use TQCGRAB_IMAGES + USE_IMAGES=true; "
            "same config on trainer and worker."
        )
    if imgs.dim() != 4:
        raise RuntimeError(
            f"Image at index {image_index} must be 4D (N,C,H,W), got {tuple(imgs.shape)}."
        )


class _GnnEffNetJointFeaturesMixin:
    """Shared feature extraction for GNN + EfficientNet + Sophy models."""

    _image_index: int
    _dim_track: int
    _use_rnn: bool
    track_gnn: nn.Module
    image_encoder: nn.Module
    img_proj: nn.Module
    physics_proj: nn.Module
    rnn: nn.GRU | None
    layernorm_joint: nn.Module

    def _joint_features(self, obs, batch_size: int) -> torch.Tensor:
        track = _ensure_float(obs[0].view(batch_size, -1)).view(batch_size, 3, self._dim_track // 3)
        physics = _obs_to_flat_tensor(obs[1 : self._image_index], batch_size)
        track_embed = self.track_gnn(track)
        imgs = _ensure_float(obs[self._image_index])
        _ensure_image_4d(imgs, self._image_index)
        img_embed = self.img_proj(self.image_encoder(imgs))
        physics_embed = self.physics_proj(physics)
        return torch.cat([track_embed, img_embed, physics_embed], dim=-1)

    def _apply_rnn(self, joint: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not self._use_rnn or self.rnn is None:
            return joint
        seq_len = int(cfg.ALG_CONFIG.get("R2D2_SEQUENCE_LENGTH", 0))
        if seq_len > 0 and batch_size % seq_len == 0:
            num_seq = batch_size // seq_len
            joint = joint.view(num_seq, seq_len, -1)
            self.rnn.flatten_parameters()
            joint, _ = self.rnn(joint)
            joint = joint.reshape(batch_size, -1)
        else:
            joint = joint.unsqueeze(1)
            self.rnn.flatten_parameters()
            joint, _ = self.rnn(joint)
            joint = joint.squeeze(1)
        return joint


class SquashedActorGnnEffNetSophyResidual(_GnnEffNetJointFeaturesMixin, TorchActorModule):
    """Sophy-style actor with GNN track encoder, EfficientNet image encoder, and residual MLP."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 8,
        embed_dim: int = 256,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
        use_frozen_effnet: bool = True,
        seed: int = cfg.SEED,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
    ):
        super().__init__(observation_space, action_space)
        torch.manual_seed(seed)
        self.use_sde = use_sde
        self._sde_clip_mean = sde_clip_mean
        spaces = _obs_spaces_list(observation_space)
        image_index = _gnn_effnet_image_index(observation_space)
        self._image_index = image_index
        dim_track, dim_physics = _gnn_effnet_physics_dims(observation_space, image_index)
        self._dim_track = dim_track
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        self.track_gnn = _build_track_gnn_branch(dim_track, hidden_dim)
        nb_channels_in = (
            int(spaces[image_index].shape[0])
            if len(spaces[image_index].shape) >= 2
            else int(prod(spaces[image_index].shape))
        )
        self.image_encoder = FrozenEfficientNetEncoder(
            nb_channels_in=nb_channels_in,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            frozen=use_frozen_effnet,
        )
        self.img_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(dim_physics, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        joint_dim = hidden_dim + embed_dim + hidden_dim
        self._use_rnn = getattr(cfg, "USE_RNN", False)
        rnn_hidden = getattr(cfg, "RNN_HIDDEN_SIZE", hidden_dim)
        self.rnn: nn.GRU | None
        if self._use_rnn:
            self.rnn = nn.GRU(joint_dim, rnn_hidden, batch_first=True)
            backbone_input_dim = rnn_hidden
        else:
            self.rnn = None
            backbone_input_dim = joint_dim
        self.layernorm_joint = nn.LayerNorm(backbone_input_dim)
        self.backbone = residual_mlp_backbone(backbone_input_dim, hidden_dim, num_blocks)
        self.head_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self._binary_brake = getattr(cfg, "BINARY_BRAKE", False) and dim_act >= 3
        self.brake_logits_layer: nn.Linear | None
        if self._binary_brake:
            self._cont_dim = 2
            self.brake_logits_layer = nn.Linear(hidden_dim, 2)
        else:
            self._cont_dim = dim_act
            self.brake_logits_layer = None

        self.mu_layer = nn.Linear(hidden_dim, self._cont_dim)
        self.sde: GSDEModule | None = None
        self.log_std_layer: nn.Linear | None = None

        if self.use_sde:
            self.sde = GSDEModule(hidden_dim, self._cont_dim, log_std_init=log_std_init)
        else:
            self.log_std_layer = nn.Linear(hidden_dim, self._cont_dim)

        self.act_limit = act_limit
        self.dropout = (
            nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])
            if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0
            else None
        )
        if dim_act > 0 and getattr(cfg, "INIT_GAS_BIAS", 0.0) != 0.0:
            with torch.no_grad():
                self.mu_layer.bias.data[0] = cfg.INIT_GAS_BIAS

    def reset_noise(self, batch_size: int = 1) -> None:
        """Sample new gSDE exploration matrix. No-op if gSDE is disabled."""
        if self.sde is not None:
            self.sde.reset_noise(batch_size)

    def load_from_bytes(self, payload: bytes, device) -> bool:
        ok = super().load_from_bytes(payload, device)
        if ok:
            self.reset_noise()
        return ok

    @staticmethod
    def _squash_log_prob(logp: torch.Tensor, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        corr = 2 * (_LOG2 - pre_tanh_action - F.softplus(-2 * pre_tanh_action))
        logp -= corr.sum(axis=1)
        return logp

    # Clamp pre-tanh to avoid -inf/NaN in squash log-prob correction when action is extreme
    _SQUASH_LOGPROB_CLAMP = 20.0

    def _policy_head_standard(self, out, mu, test, with_logprob):
        log_std = self.log_std_layer(out).float()
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()
        logp_pi = None
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            pi_action_for_corr = pi_action.clamp(
                -self._SQUASH_LOGPROB_CLAMP, self._SQUASH_LOGPROB_CLAMP
            )
            logp_pi = self._squash_log_prob(logp_pi, pi_action_for_corr)
        return torch.tanh(pi_action) * self.act_limit, logp_pi, pi_action

    def _policy_head_sde(self, out, mu, test, with_logprob):
        """Clamp pre-tanh mean to [-sde_clip_mean, sde_clip_mean] so actions never saturate.
        Forward still returns raw mu for mean_penalty gradient."""
        latent_sde = out.float()
        mu = mu.float()
        mu_clipped = torch.clamp(mu, -self._sde_clip_mean, self._sde_clip_mean)
        variance = self.sde.get_variance(latent_sde)
        pi_distribution = Normal(mu_clipped, torch.sqrt(variance + self.sde.epsilon))
        if test:
            pi_action = mu_clipped
        else:
            noise = self.sde.get_noise(latent_sde)
            pi_action = mu_clipped + noise
        logp_pi = None
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            pi_action_for_corr = pi_action.clamp(
                -self._SQUASH_LOGPROB_CLAMP, self._SQUASH_LOGPROB_CLAMP
            )
            logp_pi = self._squash_log_prob(logp_pi, pi_action_for_corr)
        return torch.tanh(pi_action) * self.act_limit, logp_pi, pi_action

    def forward(self, obs, test=False, with_logprob=True, **kwargs):
        if isinstance(obs, (tuple, list)):
            obs = list(obs)
        batch_size = obs[0].shape[0]
        joint = self._joint_features(obs, batch_size)
        joint = self._apply_rnn(joint, batch_size)
        joint = self.layernorm_joint(joint)
        out = self.backbone(joint)
        out = self.head_proj(out)
        if self.dropout is not None:
            out = self.dropout(out)

        mu = self.mu_layer(out).float()

        if self._binary_brake:
            brake_logits = self.brake_logits_layer(out).float()
            if self.use_sde and self.sde is not None:
                pi_cont, logp_cont, _ = self._policy_head_sde(out, mu, test, with_logprob)
            else:
                pi_cont, logp_cont, _ = self._policy_head_standard(out, mu, test, with_logprob)
            if test:
                brake_onehot = F.one_hot(brake_logits.argmax(dim=-1), num_classes=2).float()
            else:
                brake_onehot = F.gumbel_softmax(brake_logits, tau=1.0, hard=True)
            brake_val = brake_onehot[:, 1:2]
            # pi_cont is [gas, steer], we want [gas, brake, steer]
            pi_action = torch.cat([pi_cont[..., 0:1], brake_val, pi_cont[..., 1:2]], dim=-1)
            logp_pi = None
            if with_logprob and logp_cont is not None:
                logp_brake = F.log_softmax(brake_logits, dim=-1)
                brake_idx = (brake_val > 0.5).long().squeeze(-1)
                logp_pi = logp_cont + logp_brake.gather(1, brake_idx.unsqueeze(-1)).squeeze(-1)
            if kwargs.get("return_pre_tanh_mean", False):
                return pi_action.squeeze(), logp_pi, mu
            return pi_action.squeeze(), logp_pi
        else:
            if self.use_sde and self.sde is not None:
                pi_action, logp_pi, _ = self._policy_head_sde(out, mu, test, with_logprob)
            else:
                pi_action, logp_pi, _ = self._policy_head_standard(out, mu, test, with_logprob)
            if kwargs.get("return_pre_tanh_mean", False):
                return pi_action.squeeze(), logp_pi, mu
            return pi_action.squeeze(), logp_pi

    def act(self, obs, test=False):
        obs_seq = list(obs)
        with torch.no_grad():
            a, _ = self.forward(obs_seq, test=test, with_logprob=False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class QRCNNGnnEffNetSophyResidual(_GnnEffNetJointFeaturesMixin, nn.Module):
    """Sophy-style quantile regression critic with GNN, EfficientNet, and residual MLP."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 16,
        embed_dim: int = 256,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
        use_frozen_effnet: bool = True,
        seed: int = cfg.SEED,
    ):
        super().__init__()
        torch.manual_seed(seed)
        spaces = _obs_spaces_list(observation_space)
        image_index = _gnn_effnet_image_index(observation_space)
        self._image_index = image_index
        dim_track, dim_physics = _gnn_effnet_physics_dims(observation_space, image_index)
        self._dim_track = dim_track
        dim_act = action_space.shape[0]
        self.num_quantiles = cfo.ALG_CONFIG["QUANTILES_NUMBER"]

        self.track_gnn = _build_track_gnn_branch(dim_track, hidden_dim)
        nb_channels_in = (
            int(spaces[image_index].shape[0])
            if len(spaces[image_index].shape) >= 2
            else int(prod(spaces[image_index].shape))
        )
        self.image_encoder = FrozenEfficientNetEncoder(
            nb_channels_in=nb_channels_in,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            frozen=use_frozen_effnet,
        )
        self.img_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(dim_physics, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        joint_dim = hidden_dim + embed_dim + hidden_dim
        self._use_rnn = getattr(cfg, "USE_RNN", False)
        rnn_hidden = getattr(cfg, "RNN_HIDDEN_SIZE", hidden_dim)
        self.rnn: nn.GRU | None
        if self._use_rnn:
            self.rnn = nn.GRU(joint_dim, rnn_hidden, batch_first=True)
            backbone_input_dim = rnn_hidden
        else:
            self.rnn = None
            backbone_input_dim = joint_dim
        self.layernorm_joint = nn.LayerNorm(backbone_input_dim)
        self.backbone = residual_mlp_backbone(backbone_input_dim, hidden_dim, num_blocks)
        self.mlp_act = nn.Linear(hidden_dim + dim_act, hidden_dim)
        self.head_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        if cfg.NOISY_LINEAR_CRITIC:
            self.model_out = NoisyLinear(
                hidden_dim,
                self.num_quantiles,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                std_init=0.01,
            )
        else:
            self.model_out = nn.Linear(hidden_dim, self.num_quantiles)
        self.dropout = (
            nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])
            if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0
            else None
        )

    def forward(self, observation, act):
        if isinstance(observation, (tuple, list)):
            observation = list(observation)
        batch_size = observation[0].shape[0]
        joint = self._joint_features(observation, batch_size)
        joint = self._apply_rnn(joint, batch_size)
        joint = self.layernorm_joint(joint)
        backbone_out = self.backbone(joint)
        cat_act = torch.cat([backbone_out, act], dim=-1)
        mlp_out = F.silu(self.mlp_act(cat_act))
        out = self.head_proj(mlp_out)
        if self.dropout is not None:
            out = self.dropout(out)
        return torch.squeeze(self.model_out(out), -1)


class GnnEffNetSophyResidualActorCritic(nn.Module):
    """Complete actor-critic with GNN track + EffNet image + Sophy residual MLP."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks_actor: int = 8,
        num_blocks_critic: int = 16,
        embed_dim: int = 256,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
        use_frozen_effnet: bool = True,
        seed: int = cfg.SEED,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
    ):
        super().__init__()
        self.actor = SquashedActorGnnEffNetSophyResidual(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_actor,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            use_frozen_effnet=use_frozen_effnet,
            seed=seed,
            use_sde=use_sde,
            log_std_init=log_std_init,
            sde_clip_mean=sde_clip_mean,
        )
        self.q1 = QRCNNGnnEffNetSophyResidual(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            use_frozen_effnet=use_frozen_effnet,
            seed=seed + 1,
        )
        self.q2 = QRCNNGnnEffNetSophyResidual(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            use_frozen_effnet=use_frozen_effnet,
            seed=seed + 2,
        )

    def act(self, obs, test=False):
        with torch.no_grad():
            return self.actor.act(obs, test=test)
