"""
This module defines several neural network architectures for the Sophy-like model,
including actor and critic networks using both MLP and residual MLP backbones.
It supports TQC (Truncated Quantile Critics) and provides specialized encoders
for track information using Conv1d.
"""

import math
from io import BytesIO
from typing import cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal
from torchrl.modules import NoisyLinear

import tmrl.config as cfg
import tmrl.config.config_objects as cfo
from tmrl.actor import TorchActorModule
from tmrl.custom.models.model_blocks import residual_mlp_backbone, simba_v2_backbone
from tmrl.custom.models.model_constants import LOG_STD_MAX, LOG_STD_MIN
from tmrl.custom.utils.nn import GSDEModule


def mlp(sizes, dim_obs, activation=nn.ReLU):
    """
    Builds a multi-layer perceptron (MLP).

    Args:
        sizes (list[int]): List of layer sizes.
        dim_obs (int): Input dimension.
        activation (torch.nn.Module): Activation function class. Defaults to nn.ReLU.

    Returns:
        torch.nn.Sequential: The constructed MLP.
    """

    # Start with the input layer size
    layers = [nn.Linear(dim_obs, sizes[0]), activation()]

    # Create each layer and add to the list
    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i - 1], sizes[i]))
        layers.append(activation())

    # Build the sequential model
    model = nn.Sequential(*layers)

    return model


class QRCNNSophy(nn.Module):
    """
    Quantile Regression Critic for Sophy architecture.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes=cfg.RNN_SIZES,
        rnn_lens=cfg.RNN_LENS,
        mlp_branch_sizes=cfg.API_MLP_SIZES,
        activation=nn.ReLU,
        seed: int = cfg.SEED,
    ):
        """
        Initializes the QRCNNSophy.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            rnn_sizes: List of RNN layer sizes. Defaults to cfg.RNN_SIZES.
            rnn_lens: List of RNN lengths. Defaults to cfg.RNN_LENS.
            mlp_branch_sizes: List of sizes for MLP branches. Defaults to cfg.API_MLP_SIZES.
            activation: Activation function class. Defaults to nn.ReLU.
            seed (int): Random seed. Defaults to cfg.SEED.
        """
        super().__init__()
        torch.manual_seed(seed)

        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        self.num_quantiles = cfo.ALG_CONFIG["QUANTILES_NUMBER"]

        self.mlp_api = mlp(mlp_branch_sizes[:-1], dim_obs, activation)

        if cfg.API_LAYERNORM:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if cfg.MLP_LAYERNORM:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-2])

        self.mlp_act = mlp([mlp_branch_sizes[-1]], mlp_branch_sizes[-2] + dim_act, activation)

        self.head_proj = nn.Sequential(
            nn.Linear(mlp_branch_sizes[-1], mlp_branch_sizes[-1]),
            nn.SiLU(),
        )

        if cfg.NOISY_LINEAR_CRITIC:
            self.model_out = NoisyLinear(
                rnn_sizes[0], self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(mlp_branch_sizes[-1], self.num_quantiles)

        if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0:
            self.dropout = nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])

    def forward(self, observation, act):
        """
        Forward pass for the critic.

        Args:
            observation: Input observation.
            act: Input action.

        Returns:
            torch.Tensor: Quantile values.
        """
        batch_size = observation[0].shape[0]
        if isinstance(observation, tuple):
            observation = list(observation)

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        obs_seq_cat = torch.cat(observation, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if cfg.API_LAYERNORM:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if cfg.MLP_LAYERNORM:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        cat_mlp_api_act_out = torch.cat([mlp_api_out, act], dim=-1)

        mlp_api_out = self.mlp_act(cat_mlp_api_act_out)

        head_out = self.head_proj(mlp_api_out)

        model_out = self.model_out(head_out)

        if cfg.OUTPUT_DROPOUT > 0.0:
            model_out = self.dropout(model_out)

        return torch.squeeze(model_out, -1)


class SquashedActorSophy(TorchActorModule):
    """
    Actor network for Sophy architecture with squashed Gaussian distribution.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes=cfg.RNN_SIZES,
        rnn_lens=cfg.RNN_LENS,
        mlp_branch_sizes=cfg.API_MLP_SIZES,
        activation=nn.ReLU,
        seed: int = cfg.SEED,
    ):
        """
        Initializes the SquashedActorSophy.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            rnn_sizes: List of RNN layer sizes. Defaults to cfg.RNN_SIZES.
            rnn_lens: List of RNN lengths. Defaults to cfg.RNN_LENS.
            mlp_branch_sizes: List of sizes for MLP branches. Defaults to cfg.API_MLP_SIZES.
            activation: Activation function class. Defaults to nn.ReLU.
            seed (int): Random seed. Defaults to cfg.SEED.
        """
        super().__init__(observation_space, action_space)
        torch.manual_seed(seed)

        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        mlp_out_size = 1

        self.mlp_api = mlp(mlp_branch_sizes, dim_obs, activation)

        if cfg.API_LAYERNORM:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if cfg.MLP_LAYERNORM:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-1])

        self.head_proj = nn.Sequential(
            nn.Linear(mlp_branch_sizes[-1], mlp_branch_sizes[-1]),
            nn.SiLU(),
        )

        if cfg.NOISY_LINEAR_ACTOR:
            self.model_out = NoisyLinear(
                rnn_sizes[0], mlp_out_size, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(mlp_branch_sizes[-1], mlp_out_size)

        if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0:
            self.dropout = nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])

        self.mu_layer = nn.Linear(mlp_out_size, dim_act)
        self.log_std_layer = nn.Linear(mlp_out_size, dim_act)
        if dim_act > 0 and cfg.INIT_GAS_BIAS != 0.0:
            with torch.no_grad():
                self.mu_layer.bias.data[0] = cfg.INIT_GAS_BIAS
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX

    def forward(self, observation, test=False, with_logprob=True, **kwargs):
        """
        Forward pass for the actor.

        Args:
            observation: Input observation.
            test (bool): Whether in test mode. Defaults to False.
            with_logprob (bool): Whether to return log probability. Defaults to True.
            **kwargs: Optional return_pre_tanh_mean (bool). If True, return (action, logp_pi, mu).

        Returns:
            tuple[torch.Tensor, torch.Tensor | None] or tuple[..., torch.Tensor]: Action, log prob,
            and optionally pre-tanh mean (for L2 regularization in the trainer).
        """
        batch_size = observation[0].shape[0]
        if isinstance(observation, tuple):
            observation = list(observation)

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        obs_seq_cat = torch.cat(observation, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if cfg.API_LAYERNORM:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if cfg.MLP_LAYERNORM:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        head_out = self.head_proj(mlp_api_out)

        model_out = self.model_out(head_out)

        if cfg.OUTPUT_DROPOUT > 0.0:
            model_out = self.dropout(model_out)

        mu = self.mu_layer(model_out)
        log_std = self.log_std_layer(model_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)

        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - functional.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        pi_action = pi_action.squeeze()

        if kwargs.get("return_pre_tanh_mean", False):
            return pi_action, logp_pi, mu
        return pi_action, logp_pi

    def act(self, obs: tuple, test=False):
        """
        Predicts an action from an observation.

        Args:
            obs (tuple): Input observation.
            test (bool): Whether in test mode. Defaults to False.

        Returns:
            np.ndarray: Predicted action.
        """
        obs_seq = list(obs)
        with torch.no_grad():
            a, _ = self.forward(observation=obs_seq, test=test, with_logprob=False)
            return a.cpu().numpy()


class SophyActorCritic(nn.Module):
    """
    Actor-critic architecture for Sophy.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes=cfg.RNN_SIZES,
        rnn_lens=cfg.RNN_LENS,
        mlp_branch_sizes=cfg.API_MLP_SIZES,
        activation=nn.ReLU,
        seed: int = cfg.SEED,
    ):
        """
        Initializes the SophyActorCritic.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            rnn_sizes: List of RNN layer sizes. Defaults to cfg.RNN_SIZES.
            rnn_lens: List of RNN lengths. Defaults to cfg.RNN_LENS.
            mlp_branch_sizes: List of sizes for MLP branches. Defaults to cfg.API_MLP_SIZES.
            activation: Activation function class. Defaults to nn.ReLU.
            seed (int): Random seed. Defaults to cfg.SEED.
        """
        super().__init__()
        self.actor = SquashedActorSophy(
            observation_space,
            action_space,
            rnn_sizes,
            rnn_lens,
            mlp_branch_sizes,
            activation,
            seed=seed,
        )
        self.q1 = QRCNNSophy(
            observation_space,
            action_space,
            rnn_sizes,
            rnn_lens,
            mlp_branch_sizes,
            activation,
            seed=seed + 1,
        )
        self.q2 = QRCNNSophy(
            observation_space,
            action_space,
            rnn_sizes,
            rnn_lens,
            mlp_branch_sizes,
            activation,
            seed=seed + 2,
        )


def _build_track_conv1d_branch(dim_track: int, hidden_dim: int) -> nn.Module:
    """
    Builds a Conv1d extractor for track_info.

    Args:
        dim_track (int): Dimension of the track information.
        hidden_dim (int): Dimension of the output hidden layer.

    Returns:
        torch.nn.Module: The constructed Conv1d branch.
    """
    assert dim_track >= 3, "track dim must be at least 3"
    assert dim_track % 3 == 0, "track dim must be 6*N (3 channels of 2*N)"
    return nn.Sequential(
        nn.Conv1d(3, 32, kernel_size=5, padding=2),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(64, hidden_dim),
    )


def _build_track_spline_mlp_branch(dim_track: int, hidden_dim: int) -> nn.Module:
    """
    Parametric track encoder: compresses track (left/center/right) into a low-dim
    representation via pooling + MLP (B-spline / Frenet-Serret style compact encoding).
    Input (B, 3, N) -> pool to fixed size -> MLP -> (B, hidden_dim).
    """
    assert dim_track >= 3, "track dim must be at least 3"
    assert dim_track % 3 == 0, "track dim must be 3*N (3 channels)"
    n_pts = dim_track // 3
    pool_size = min(16, max(4, n_pts // 4))
    return nn.Sequential(
        nn.AdaptiveAvgPool1d(pool_size),
        nn.Flatten(),
        nn.Linear(3 * pool_size, 128),
        nn.LayerNorm(128),
        nn.SiLU(),
        nn.Linear(128, hidden_dim),
    )


class _TrackGNN(nn.Module):
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
        for lin, norm in zip(self.layers, self.norms, strict=False):
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


def _make_backbone(input_dim: int, hidden_dim: int, num_blocks: int) -> nn.Module:
    """Build residual MLP or SimbaV2 backbone depending on config."""
    if getattr(cfg, "USE_SIMBAV2", False):
        return simba_v2_backbone(input_dim, hidden_dim, num_blocks)
    return residual_mlp_backbone(input_dim, hidden_dim, num_blocks)


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


class SquashedActorSophyResidual(TorchActorModule):
    """
    Sophy actor with residual MLP backbone and optional Conv1d track branch and RNN.
    Supports gSDE (generalized State-Dependent Exploration) when use_sde=True.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim=cfg.RESIDUAL_MLP_HIDDEN_DIM,
        num_blocks=cfg.RESIDUAL_MLP_NUM_BLOCKS,
        seed: int = cfg.SEED,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
    ):
        super().__init__(observation_space, action_space)
        torch.manual_seed(seed)

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self._dim_obs = dim_obs
        self.use_sde = use_sde
        self._sde_clip_mean = sde_clip_mean
        self._use_track_conv = getattr(cfg, "USE_TRACK_CONV1D", True) and len(observation_space) > 1
        if self._use_track_conv:
            dim_track_first = math.prod(observation_space[0].shape)
            if dim_track_first % 3 != 0:
                # Ego-only space (e.g. asymmetric actor): first slot is not track
                self._use_track_conv = False
        self._use_rnn = getattr(cfg, "USE_RNN", False)
        rnn_hidden = getattr(cfg, "RNN_HIDDEN_SIZE", hidden_dim)

        if self._use_track_conv:
            dim_track = math.prod(observation_space[0].shape)
            dim_physics = dim_obs - dim_track
            assert dim_track % 3 == 0, "track_info should be 6*N (left+center+right)"
            self._dim_track = dim_track
            self._dim_physics = dim_physics
            track_encoder = getattr(cfg, "TRACK_ENCODER", "conv1d")
            if track_encoder == "spline_mlp":
                self.track_conv = _build_track_spline_mlp_branch(dim_track, hidden_dim)
            elif track_encoder == "gnn":
                self.track_conv = _build_track_gnn_branch(dim_track, hidden_dim)
            else:
                self.track_conv = _build_track_conv1d_branch(dim_track, hidden_dim)
            self.physics_proj = nn.Sequential(
                nn.Linear(dim_physics, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            joint_dim = 2 * hidden_dim
            self.rnn: nn.GRU | None = None
            if self._use_rnn:
                self.rnn = nn.GRU(joint_dim, rnn_hidden, batch_first=True)
                backbone_input_dim = rnn_hidden
            else:
                backbone_input_dim = joint_dim
            self.layernorm_joint = nn.LayerNorm(backbone_input_dim)
            self.backbone = _make_backbone(backbone_input_dim, hidden_dim, num_blocks)
            self.layernorm_api = None
        else:
            self.layernorm_api = nn.LayerNorm(dim_obs) if cfg.API_LAYERNORM else None
            self.backbone = _make_backbone(dim_obs, hidden_dim, num_blocks)

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

        if dim_act > 0 and cfg.INIT_GAS_BIAS != 0.0:
            with torch.no_grad():
                self.mu_layer.bias.data[0] = cfg.INIT_GAS_BIAS
        self.act_limit = act_limit
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self._dim_act = dim_act
        self.dropout = (
            nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])
            if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0
            else None
        )

    def reset_noise(self, batch_size: int = 1) -> None:
        """Sample new gSDE exploration matrix. No-op if gSDE is disabled."""
        if self.sde is not None:
            self.sde.reset_noise(batch_size)

    def load_from_bytes(self, payload: bytes, device) -> bool:
        ok = super().load_from_bytes(payload, device)
        if ok:
            self.reset_noise()
        return ok

    def _joint_features(self, observation, batch_size: int) -> torch.Tensor:
        """
        Combines track features from Conv1d and physics features from MLP.

        Args:
            observation: Input observation.
            batch_size (int): Current batch size.

        Returns:
            torch.Tensor: Combined features.
        """
        track = observation[0].view(batch_size, -1).float()
        track = track.view(batch_size, 3, self._dim_track // 3)
        track_embed = self.track_conv(track)
        physics = _obs_to_flat_tensor(observation[1:], batch_size)
        physics_embed = self.physics_proj(physics)
        return torch.cat([track_embed, physics_embed], dim=-1)

    def _obs_to_tensor(self, observation):
        """
        Converts observation to a single tensor.

        Args:
            observation: Input observation.

        Returns:
            torch.Tensor: Combined observation tensor.
        """
        if isinstance(observation, torch.Tensor):
            return observation.view(observation.shape[0], -1).float()
        if isinstance(observation, (tuple, list)) and len(observation) > 0:
            batch_size = observation[0].shape[0]
            return _obs_to_flat_tensor(observation, batch_size)
        raise ValueError(
            "SophyResidual actor expected observation to be a non-empty tuple/list of tensors "
            "or a single tensor (batch, dim). Got empty or unsupported type."
        )

    def _compute_features(self, observation):
        """Extract backbone features from observation."""
        if isinstance(observation, (tuple, list)):
            observation = list(observation)
        batch_size = observation[0].shape[0]

        if self._use_track_conv:
            joint = self._joint_features(observation, batch_size)
            if self._use_rnn and self.rnn is not None:
                seq_len = int(cfg.ALG_CONFIG.get("R2D2_SEQUENCE_LENGTH", 0))

                # --- FIX: R2D2 Burn-in Protocol ---
                # If using RNN, we bifurcate the sequence into Burn-In (Detached)
                # and Active Learning (BPTT).
                burn_in_len = int(cfg.ALG_CONFIG.get("R2D2_BURN_IN", 0))

                if (
                    seq_len > 0
                    and batch_size % seq_len == 0
                    and burn_in_len > 0
                    and burn_in_len < seq_len
                ):
                    num_seq = batch_size // seq_len
                    joint_seq = joint.view(num_seq, seq_len, -1)

                    # Stage 1: Hidden State Recovery (No Gradients)
                    # Burn-in no_grad to avoid gradient leak; recover hidden from current weights.
                    with torch.no_grad():
                        # Process first B steps to correct representational drift
                        joint_burn = joint_seq[:, :burn_in_len, :]
                        self.rnn.flatten_parameters()
                        _, h_burn = self.rnn(joint_burn)

                    # Stage 2: Active Learning (Attach Gradients)
                    joint_active = joint_seq[:, burn_in_len:, :]
                    self.rnn.flatten_parameters()
                    out_active, _ = self.rnn(joint_active, h_burn)

                    # Need to return full sequence size to match actions/rewards
                    # So we pad the burn_in part with detached outputs or zeros
                    with torch.no_grad():
                        out_burn, _ = self.rnn(joint_burn)

                    joint = torch.cat([out_burn, out_active], dim=1).reshape(batch_size, -1)
                elif seq_len > 0 and batch_size % seq_len == 0:
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
            joint = self.layernorm_joint(joint)
            backbone_out = self.backbone(joint)
        else:
            obs_seq_cat = self._obs_to_tensor(observation)
            if obs_seq_cat.shape[-1] != self._dim_obs:
                raise ValueError(
                    f"SophyResidual actor expected observation dimension {self._dim_obs}, "
                    f"but got {obs_seq_cat.shape[-1]}."
                )
            if self.layernorm_api is not None:
                obs_seq_cat = self.layernorm_api(obs_seq_cat)
            backbone_out = self.backbone(obs_seq_cat)

        out = self.head_proj(backbone_out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    @staticmethod
    def _squash_log_prob(logp: torch.Tensor, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Apply tanh squash correction to Gaussian log prob (SAC appendix C)."""
        corr = 2 * (np.log(2) - pre_tanh_action - functional.softplus(-2 * pre_tanh_action))
        logp -= corr.sum(axis=1)
        return logp

    def _policy_head_standard(self, out, mu, test, with_logprob):
        """Standard Gaussian + tanh policy head."""
        log_std = self.log_std_layer(out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()
        logp_pi = None
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi = self._squash_log_prob(logp_pi, pi_action)
        return torch.tanh(pi_action) * self.act_limit, logp_pi, pi_action

    def _policy_head_sde(self, out, mu, test, with_logprob):
        """gSDE policy head: state-dependent noise through exploration matrix.
        Clamp pre-tanh mean to [-sde_clip_mean, sde_clip_mean] so actions never saturate
        (tanh(±2) ≈ ±0.96). Forward still returns raw mu for mean_penalty gradient.
        """
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
            logp_pi = self._squash_log_prob(logp_pi, pi_action)
        return torch.tanh(pi_action) * self.act_limit, logp_pi, pi_action

    def forward(self, observation, test=False, with_logprob=True, **kwargs):
        out = self._compute_features(observation)
        mu = self.mu_layer(out)

        if self._binary_brake:
            brake_logits = self.brake_logits_layer(out).float()
            if self.use_sde and self.sde is not None:
                pi_cont, logp_cont, _ = self._policy_head_sde(out, mu, test, with_logprob)
            else:
                pi_cont, logp_cont, _pre_tanh = self._policy_head_standard(
                    out, mu, test, with_logprob
                )
            if test:
                brake_onehot = functional.one_hot(
                    brake_logits.argmax(dim=-1), num_classes=2
                ).float()
            else:
                brake_onehot = functional.gumbel_softmax(brake_logits, tau=1.0, hard=True)
            brake_val = brake_onehot[:, 1:2]
            # pi_cont is [gas, steer], we want [gas, brake, steer]
            pi_action = torch.cat([pi_cont[..., 0:1], brake_val, pi_cont[..., 1:2]], dim=-1)
            logp_pi = None
            if with_logprob and logp_cont is not None:
                logp_brake = functional.log_softmax(brake_logits, dim=-1)
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
        """
        Predicts an action from an observation.

        Args:
            obs: Input observation.
            test (bool): Whether in test mode. Defaults to False.

        Returns:
            np.ndarray: Predicted action.
        """
        obs_seq = list(obs)
        with torch.no_grad():
            a, _ = self.forward(observation=obs_seq, test=test, with_logprob=False)
            return a.cpu().numpy()


class QRCNNSophyResidual(nn.Module):
    """
    Sophy critic (TQC quantiles) with residual MLP backbone and optional
    Conv1d track branch and RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim=cfg.RESIDUAL_MLP_HIDDEN_DIM,
        num_blocks=cfg.RESIDUAL_MLP_NUM_BLOCKS,
        seed: int = cfg.SEED,
    ):
        """
        Initializes the QRCNNSophyResidual.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            hidden_dim (int): Hidden dimension for MLP. Defaults to cfg.RESIDUAL_MLP_HIDDEN_DIM.
            num_blocks (int): Number of residual blocks. Defaults to cfg.RESIDUAL_MLP_NUM_BLOCKS.
            seed (int): Random seed. Defaults to cfg.SEED.
        """
        super().__init__()
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        self.num_quantiles = cfo.ALG_CONFIG["QUANTILES_NUMBER"]

        self._use_track_conv = getattr(cfg, "USE_TRACK_CONV1D", True) and len(observation_space) > 1
        if self._use_track_conv:
            dim_track_first = math.prod(observation_space[0].shape)
            if dim_track_first % 3 != 0:
                self._use_track_conv = False
        self._use_rnn = getattr(cfg, "USE_RNN", False)
        rnn_hidden = getattr(cfg, "RNN_HIDDEN_SIZE", hidden_dim)

        if self._use_track_conv:
            dim_track = math.prod(observation_space[0].shape)
            dim_physics = dim_obs - dim_track
            assert dim_track % 3 == 0, "track_info should be 6*N"
            self._dim_track = dim_track
            self._dim_physics = dim_physics
            track_encoder = getattr(cfg, "TRACK_ENCODER", "conv1d")
            if track_encoder == "spline_mlp":
                self.track_conv = _build_track_spline_mlp_branch(dim_track, hidden_dim)
            elif track_encoder == "gnn":
                self.track_conv = _build_track_gnn_branch(dim_track, hidden_dim)
            else:
                self.track_conv = _build_track_conv1d_branch(dim_track, hidden_dim)
            self.physics_proj = nn.Sequential(
                nn.Linear(dim_physics, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            joint_dim = 2 * hidden_dim
            self.rnn: nn.GRU | None = None
            if self._use_rnn:
                self.rnn = nn.GRU(joint_dim, rnn_hidden, batch_first=True)
                backbone_input_dim = rnn_hidden
            else:
                backbone_input_dim = joint_dim
            self.layernorm_joint = nn.LayerNorm(backbone_input_dim)
            self.backbone = _make_backbone(backbone_input_dim, hidden_dim, num_blocks)
            self.layernorm_api = None
        else:
            self.layernorm_api = nn.LayerNorm(dim_obs) if cfg.API_LAYERNORM else None
            self.backbone = _make_backbone(dim_obs, hidden_dim, num_blocks)

        self.mlp_act = nn.Linear(hidden_dim + dim_act, hidden_dim)
        self.head_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        if cfg.NOISY_LINEAR_CRITIC:
            self.model_out = NoisyLinear(
                hidden_dim, self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(hidden_dim, self.num_quantiles)
        self.dropout = (
            nn.Dropout(cfg.MODEL_CONFIG["OUTPUT_DROPOUT"])
            if cfg.MODEL_CONFIG["OUTPUT_DROPOUT"] > 0.0
            else None
        )

    def _joint_features(self, observation, batch_size: int) -> torch.Tensor:
        """
        Combines track features from Conv1d and physics features from MLP.

        Args:
            observation: Input observation.
            batch_size (int): Current batch size.

        Returns:
            torch.Tensor: Combined features.
        """
        if self._use_track_conv:
            track = observation[0].view(batch_size, -1).float()
            track = track.view(batch_size, 3, self._dim_track // 3)
            track_embed = self.track_conv(track)
            physics = _obs_to_flat_tensor(observation[1:], batch_size)
            physics_embed = self.physics_proj(physics)
            return torch.cat([track_embed, physics_embed], dim=-1)
        result: torch.Tensor = self._obs_to_tensor(observation)
        return result

    def _obs_to_tensor(self, observation) -> torch.Tensor:
        """
        Converts observation to a single tensor.

        Args:
            observation: Input observation.

        Returns:
            torch.Tensor: Combined observation tensor.
        """
        if isinstance(observation, torch.Tensor):
            return observation.view(observation.shape[0], -1).float()
        if isinstance(observation, (tuple, list)) and len(observation) > 0:
            batch_size = observation[0].shape[0]
            return _obs_to_flat_tensor(observation, batch_size)
        raise ValueError(
            "SophyResidual critic expected observation to be a non-empty tuple/list of tensors "
            "or a single tensor (batch, dim). Got empty or unsupported type."
        )

    def forward(self, observation, act):
        """
        Forward pass for the critic.

        Args:
            observation: Input observation.
            act: Input action.

        Returns:
            torch.Tensor: Quantile values.
        """
        if isinstance(observation, (tuple, list)):
            observation = list(observation)
        batch_size = observation[0].shape[0]

        if self._use_track_conv:
            joint = self._joint_features(observation, batch_size)
            if self._use_rnn and self.rnn is not None:
                seq_len = int(cfg.ALG_CONFIG.get("R2D2_SEQUENCE_LENGTH", 0))

                # --- FIX: R2D2 Burn-in Protocol ---
                # If using RNN, we bifurcate the sequence into Burn-In (Detached)
                # and Active Learning (BPTT).
                burn_in_len = int(cfg.ALG_CONFIG.get("R2D2_BURN_IN", 0))

                if (
                    seq_len > 0
                    and batch_size % seq_len == 0
                    and burn_in_len > 0
                    and burn_in_len < seq_len
                ):
                    num_seq = batch_size // seq_len
                    joint_seq = joint.view(num_seq, seq_len, -1)

                    # Stage 1: Hidden State Recovery (No Gradients)
                    # Burn-in no_grad to avoid gradient leak; recover hidden from current weights.
                    with torch.no_grad():
                        # Process first B steps to correct representational drift
                        joint_burn = joint_seq[:, :burn_in_len, :]
                        self.rnn.flatten_parameters()
                        _, h_burn = self.rnn(joint_burn)

                    # Stage 2: Active Learning (Attach Gradients)
                    joint_active = joint_seq[:, burn_in_len:, :]
                    self.rnn.flatten_parameters()
                    out_active, _ = self.rnn(joint_active, h_burn)

                    # Need to return full sequence size to match actions/rewards
                    # So we pad the burn_in part with detached outputs or zeros
                    with torch.no_grad():
                        out_burn, _ = self.rnn(joint_burn)

                    joint = torch.cat([out_burn, out_active], dim=1).reshape(batch_size, -1)
                elif seq_len > 0 and batch_size % seq_len == 0:
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
            joint = self.layernorm_joint(joint)
            backbone_out = self.backbone(joint)
        else:
            obs_seq_cat = self._obs_to_tensor(observation)
            if self.layernorm_api is not None:
                obs_seq_cat = self.layernorm_api(obs_seq_cat)
            backbone_out = self.backbone(obs_seq_cat)

        cat_act = torch.cat([backbone_out, act], dim=-1)
        mlp_out = torch.nn.functional.silu(self.mlp_act(cat_act))
        out = self.head_proj(mlp_out)
        if self.dropout is not None:
            out = self.dropout(out)
        return torch.squeeze(self.model_out(out), -1)


class SophyResidualActorCritic(nn.Module):
    """
    Actor-critic for TQC with residual MLP backbone (LayerNorm + SiLU).
    Asymmetric: critic can have more blocks than actor.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim=cfg.RESIDUAL_MLP_HIDDEN_DIM,
        num_blocks_actor=cfg.RESIDUAL_MLP_NUM_BLOCKS_ACTOR,
        num_blocks_critic=cfg.RESIDUAL_MLP_NUM_BLOCKS_CRITIC,
        seed: int = cfg.SEED,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
    ):
        """
        Initializes the SophyResidualActorCritic.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            hidden_dim (int): Hidden dimension for MLP. Defaults to cfg.RESIDUAL_MLP_HIDDEN_DIM.
            num_blocks_actor (int): Number of residual blocks for actor.
                Defaults to cfg.RESIDUAL_MLP_NUM_BLOCKS_ACTOR.
            num_blocks_critic (int): Number of residual blocks for critic.
                Defaults to cfg.RESIDUAL_MLP_NUM_BLOCKS_CRITIC.
            seed (int): Random seed. Defaults to cfg.SEED.
            use_sde (bool): Enable generalized State-Dependent Exploration.
            log_std_init (float): Initial log-std for gSDE.
            sde_clip_mean (float): Clip pre-tanh mean when using gSDE.
        """
        super().__init__()
        self.actor = SquashedActorSophyResidual(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_actor,
            seed=seed,
            use_sde=use_sde,
            log_std_init=log_std_init,
            sde_clip_mean=sde_clip_mean,
        )
        # Critic receives full privileged observation (Track + Telemetry)
        self.q1 = QRCNNSophyResidual(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            seed=seed + 1,
        )
        self.q2 = QRCNNSophyResidual(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            seed=seed + 2,
        )

    def act(self, obs, test=False):
        """
        Predicts an action from an observation.

        Args:
            obs: Input observation.
            test (bool): Whether in test mode. Defaults to False.

        Returns:
            np.ndarray: Predicted action.
        """
        with torch.no_grad():
            return self.actor.act(obs, test=test)


class _AsymmetricActorAdapter(TorchActorModule):
    """Adapter so actor can consume full obs and internally keep ego-only inputs."""

    def __init__(self, actor: SquashedActorSophyResidual, full_obs_len: int):
        super().__init__(actor.observation_space, actor.action_space)
        self.actor = actor
        self.full_obs_len = full_obs_len

    def _to_ego_obs(self, obs):
        if (
            isinstance(obs, (tuple, list))
            and self.full_obs_len > 1
            and len(obs) == self.full_obs_len
        ):
            return obs[1:]
        return obs

    def forward(self, observation, test=False, with_logprob=True, **kwargs):
        return self.actor(
            self._to_ego_obs(observation), test=test, with_logprob=with_logprob, **kwargs
        )

    def act(self, obs, test=False):
        return self.actor.act(self._to_ego_obs(obs), test=test)

    def save_to_bytes(self) -> bytes:
        """Serialize only the inner actor so the worker (a bare SquashedActorSophyResidual)
        can load the state_dict without an 'actor.' key prefix."""
        buffer = BytesIO()
        torch.save(self.actor.state_dict(), buffer)
        return buffer.getvalue()

    def load_from_bytes(self, payload: bytes, device) -> bool:
        self.device = device
        buffer = BytesIO(payload)
        try:
            state = torch.load(buffer, map_location=self.device, weights_only=True)
            self.actor.load_state_dict(state)
        except RuntimeError as e:
            err = str(e)
            if "size mismatch" in err or "Missing key" in err or "shape" in err.lower():
                from loguru import logger

                logger.warning(
                    "Ignoring incompatible asymmetric actor weights (shape mismatch): {}",
                    err.split("\n", 1)[0].strip(),
                )
                return False
            raise
        if hasattr(self.actor, "reset_noise"):
            self.actor.reset_noise()
        return True

    def reset_noise(self, batch_size: int = 1) -> None:
        if hasattr(self.actor, "reset_noise"):
            self.actor.reset_noise(batch_size)


# ASYMMETRIC EGO/GLOBAL SEPARATOR
class AsymmetricSophyResidualActorCritic(nn.Module):
    """
    Implements the Blueprint from GT Sophy:
    Actor: Restricted to ego-centric telemetry (velocity, inputs, local rays).
    Critic: Privileged access to global track geometry lookahead.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim=cfg.RESIDUAL_MLP_HIDDEN_DIM,
        num_blocks_actor=cfg.RESIDUAL_MLP_NUM_BLOCKS_ACTOR,
        num_blocks_critic=cfg.RESIDUAL_MLP_NUM_BLOCKS_CRITIC,
        seed: int = cfg.SEED,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
    ):
        super().__init__()
        # Actor only receives ego-centric telemetry (drop privileged track slot 0).
        if hasattr(observation_space, "spaces"):
            ego_space = tuple(observation_space.spaces[1:])
        elif hasattr(observation_space, "__getitem__"):
            ego_space = observation_space[1:]
        else:
            ego_space = observation_space

        base_actor = SquashedActorSophyResidual(
            observation_space=ego_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_actor,
            seed=seed,
            use_sde=use_sde,
            log_std_init=log_std_init,
            sde_clip_mean=sde_clip_mean,
        )
        full_obs_len = len(observation_space) if hasattr(observation_space, "__len__") else 0
        self.actor = _AsymmetricActorAdapter(base_actor, full_obs_len)

        # Critic receives full privileged observation (Track + Telemetry)
        self.q1 = QRCNNSophyResidual(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            seed=seed + 1,
        )
        self.q2 = QRCNNSophyResidual(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            seed=seed + 2,
        )

    def act(self, obs, test=False):
        # Slice privileged data off for the actor (indices 1:15 to be safe with tuple spaces)
        if (isinstance(obs, tuple) and len(obs) > 1) or (isinstance(obs, list) and len(obs) > 1):
            ego_obs = obs[1:]
        else:
            ego_obs = obs
        return self.actor.act(ego_obs, test=test)
