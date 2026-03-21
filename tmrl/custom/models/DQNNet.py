"""IQN Q-network with Dueling heads for discrete-action DQN.

Reuses the same track encoder (GNN / Conv1d / spline-MLP) and residual-MLP
backbone that the continuous-action Sophy models use, but replaces the
actor/critic heads with:
  - Implicit Quantile Network (IQN) cosine embedding for distributional RL
  - Dueling architecture (V + A streams)

The network outputs Q(s, a) for every discrete action in a single forward pass.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tmrl.config as cfg
from tmrl.actor import TorchActorModule
from tmrl.custom.models.model_blocks import (
    residual_mlp_backbone,
    simba_v2_backbone,
)
from tmrl.custom.models.Sophy import (
    _build_track_conv1d_branch,
    _build_track_gnn_branch,
    _build_track_spline_mlp_branch,
    _obs_to_flat_tensor,
)


class CosineEmbedding(nn.Module):
    """Cosine basis embedding for IQN quantile fractions.

    Maps tau in [0, 1] to a fixed-size vector using:
        phi(tau)_j = ReLU(sum_i cos(pi * i * tau) * w_ij + b_j)
    """

    def __init__(self, n_cos: int = 64, embed_dim: int = 64):
        super().__init__()
        self.n_cos = n_cos
        self.linear = nn.Linear(n_cos, embed_dim)
        i_pi = torch.arange(1, n_cos + 1, dtype=torch.float32) * math.pi
        self.register_buffer("i_pi", i_pi)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Map quantile fractions to embeddings.

        Args:
            tau: Quantile fractions of shape ``(batch, n_quantiles)``.

        Returns:
            Embeddings of shape ``(batch, n_quantiles, embed_dim)``.
        """
        from einops import rearrange

        i_pi: torch.Tensor = self.i_pi  # type: ignore[assignment]
        cos_input = rearrange(tau, "b n -> b n 1") * i_pi
        cos_features = torch.cos(cos_input)
        return F.relu(self.linear(cos_features))


class IQNFeatureBackbone(nn.Module):
    """Shared feature extractor for the IQN Q-network.

    Encodes the observation (track + physics telemetry) into a fixed-dim
    feature vector, identical to the Sophy actor backbone, then multiplies
    element-wise with the cosine-embedded quantile fractions.
    """

    def __init__(
        self,
        observation_space,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        n_cos: int = 64,
    ):
        super().__init__()
        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        self._use_track_conv = getattr(cfg, "USE_TRACK_CONV1D", True) and len(observation_space) > 1
        if self._use_track_conv:
            dim_track_first = math.prod(observation_space[0].shape)
            if dim_track_first % 3 != 0:
                self._use_track_conv = False

        if self._use_track_conv:
            dim_track = math.prod(observation_space[0].shape)
            dim_physics = dim_obs - dim_track
            self._dim_track = dim_track
            track_encoder_name = getattr(cfg, "TRACK_ENCODER", "conv1d")
            if track_encoder_name == "spline_mlp":
                self.track_conv = _build_track_spline_mlp_branch(dim_track, hidden_dim)
            elif track_encoder_name == "gnn":
                self.track_conv = _build_track_gnn_branch(dim_track, hidden_dim)
            else:
                self.track_conv = _build_track_conv1d_branch(dim_track, hidden_dim)
            self.physics_proj = nn.Sequential(
                nn.Linear(dim_physics, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            joint_dim = 2 * hidden_dim
            self._use_rnn = bool(getattr(cfg, "USE_RNN_MODEL", False))
            rnn_hidden = int(getattr(cfg, "RNN_HIDDEN_SIZE", hidden_dim))
            self.rnn: nn.GRU | None
            if self._use_rnn:
                self.rnn = nn.GRU(joint_dim, rnn_hidden, num_layers=1, batch_first=True)
                backbone_input_dim = rnn_hidden
            else:
                self.rnn = None
                backbone_input_dim = joint_dim
            self.layernorm_joint = nn.LayerNorm(backbone_input_dim)
            self.layernorm_api = None
        else:
            self._dim_track = 0
            self.rnn = None
            self._use_rnn = False
            self.layernorm_api = nn.LayerNorm(dim_obs) if cfg.API_LAYERNORM else None
            backbone_input_dim = dim_obs

        self.backbone: nn.Module
        if getattr(cfg, "USE_SIMBAV2", False):
            self.backbone = simba_v2_backbone(backbone_input_dim, hidden_dim, num_blocks)
        else:
            self.backbone = residual_mlp_backbone(backbone_input_dim, hidden_dim, num_blocks)
        self.cos_embed = CosineEmbedding(n_cos=n_cos, embed_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def _joint_features(self, observation, batch_size: int) -> torch.Tensor:
        track = observation[0].view(batch_size, -1).float()
        track = track.view(batch_size, 3, self._dim_track // 3)
        track_embed = self.track_conv(track)
        physics = _obs_to_flat_tensor(observation[1:], batch_size)
        physics_embed = self.physics_proj(physics)
        return torch.cat([track_embed, physics_embed], dim=-1)

    def _gru_joint(self, joint: torch.Tensor) -> torch.Tensor:
        """Single-layer GRU on track+physics joint (matches SophyResidual)."""
        if self.rnn is None:
            return joint
        batch_size = joint.shape[0]
        seq_len = int(cfg.ALG_CONFIG.get("R2D2_SEQUENCE_LENGTH", 0))
        burn_in_len = int(cfg.ALG_CONFIG.get("R2D2_BURN_IN", 0))

        if seq_len > 0 and batch_size % seq_len == 0 and burn_in_len > 0 and burn_in_len < seq_len:
            num_seq = batch_size // seq_len
            joint_seq = joint.view(num_seq, seq_len, -1)
            with torch.no_grad():
                joint_burn = joint_seq[:, :burn_in_len, :]
                self.rnn.flatten_parameters()
                _, h_burn = self.rnn(joint_burn)
            joint_active = joint_seq[:, burn_in_len:, :]
            self.rnn.flatten_parameters()
            out_active, _ = self.rnn(joint_active, h_burn)
            with torch.no_grad():
                out_burn, _ = self.rnn(joint_burn)
            joint_out = torch.cat([out_burn, out_active], dim=1).reshape(batch_size, -1)
            return joint_out
        if seq_len > 0 and batch_size % seq_len == 0:
            num_seq = batch_size // seq_len
            joint_seq = joint.view(num_seq, seq_len, -1)
            self.rnn.flatten_parameters()
            joint_out, _ = self.rnn(joint_seq)
            joint_out_seq: torch.Tensor = joint_out
            return joint_out_seq.reshape(batch_size, -1)
        joint_seq = joint.unsqueeze(1)
        self.rnn.flatten_parameters()
        joint_out, _ = self.rnn(joint_seq)
        joint_out_single: torch.Tensor = joint_out
        return joint_out_single.squeeze(1)

    def forward(self, observation, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: tuple of tensors from the env.
            tau: (batch, n_quantiles) sampled quantile fractions.

        Returns:
            (batch, n_quantiles, hidden_dim) features ready for the heads.
        """
        if isinstance(observation, (tuple, list)):
            observation = list(observation)
        batch_size = observation[0].shape[0]

        if self._use_track_conv:
            joint = self._joint_features(observation, batch_size)
            joint = self._gru_joint(joint)
            joint = self.layernorm_joint(joint)
            features = self.backbone(joint)
        else:
            obs_flat = _obs_to_flat_tensor(observation, batch_size)
            if self.layernorm_api is not None:
                obs_flat = self.layernorm_api(obs_flat)
            features = self.backbone(obs_flat)

        from einops import rearrange

        tau_embed = self.cos_embed(tau)
        combined: torch.Tensor = rearrange(features, "b h -> b 1 h") * tau_embed
        return combined


class DuelingHead(nn.Module):
    """Dueling DQN head: Q(s,a) = V(s) + A(s,a) - mean(A)."""

    def __init__(self, hidden_dim: int, n_actions: int):
        super().__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map features to Q-values via dueling decomposition.

        Args:
            features: Tensor of shape ``(..., hidden_dim)``.

        Returns:
            Q-values of shape ``(..., n_actions)``.
        """
        v = self.value_stream(features)
        a = self.advantage_stream(features)
        result: torch.Tensor = v + a - a.mean(dim=-1, keepdim=True)
        return result


class IQNQNetwork(nn.Module):
    """Full IQN Q-network with optional Dueling architecture.

    Forward pass samples n_quantiles tau values, embeds them, and produces
    per-action quantile values of shape (batch, n_quantiles, n_actions).
    The mean over quantiles gives the expected Q-values.
    """

    def __init__(
        self,
        observation_space,
        n_actions: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        n_cos: int = 64,
        dueling: bool = True,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.backbone = IQNFeatureBackbone(
            observation_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            n_cos=n_cos,
        )
        self.head: nn.Module
        if dueling:
            self.head = DuelingHead(hidden_dim, n_actions)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, n_actions),
            )

    def forward(
        self,
        observation,
        tau: torch.Tensor | None = None,
        n_quantiles: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observation: env observation tuple.
            tau: (batch, n_quantiles) fractions. Sampled if None.
            n_quantiles: how many quantiles to sample when tau is None.

        Returns:
            quantile_values: (batch, n_quantiles, n_actions)
            tau: the quantile fractions used.
        """
        batch_size = observation[0].shape[0]
        device = observation[0].device

        if tau is None:
            tau = torch.rand(batch_size, n_quantiles, device=device)

        features = self.backbone(observation, tau)
        quantile_values: torch.Tensor = self.head(features)
        return quantile_values, tau

    def q_values(self, observation, n_quantiles: int = 32) -> torch.Tensor:
        """Expected Q-values: mean over quantile dimension.

        Returns:
            (batch, n_actions)
        """
        qv, _ = self.forward(observation, n_quantiles=n_quantiles)
        return qv.mean(dim=1)


class DQNActor(TorchActorModule):
    """Actor for DQN rollout workers: wraps IQNQNetwork with epsilon-greedy.

    The trainer broadcasts updated Q-network weights to this actor.
    At inference, it computes Q-values and selects actions epsilon-greedily.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        n_cos: int = 64,
        dueling: bool = True,
        n_actions: int = 78,
        epsilon: float = 0.00005,
        n_quantiles_eval: int = 32,
        explore_repeat_steps: int = 1,
    ):
        super().__init__(observation_space, action_space)
        self.q_net = IQNQNetwork(
            observation_space,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            n_cos=n_cos,
            dueling=dueling,
        )
        # Store epsilon as a buffer so it is included in state_dict and
        # survives save_to_bytes/load_from_bytes serialization.
        self.register_buffer("_epsilon_buf", torch.tensor(epsilon, dtype=torch.float32))
        self.n_actions = n_actions
        self.n_quantiles_eval = n_quantiles_eval
        self.explore_repeat_steps = max(1, int(explore_repeat_steps))
        self._current_explore_count = 0
        self._last_explore_action: np.ndarray | None = None

    @property
    def epsilon(self):
        return self._epsilon_buf.item()

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon_buf.fill_(value)

    def forward(self, observation, **kwargs):
        return self.q_net.q_values(observation, n_quantiles=self.n_quantiles_eval)

    def act_(self, obs, test=False):
        """Override base act_ to skip the float np.clip that breaks integer actions."""
        from tmrl.util import collate_torch

        obs = collate_torch([obs], device=self.device)
        with torch.no_grad():
            action = self.act(obs, test=test)
        return action

    def act(self, obs, test=False):
        """Epsilon-greedy action selection.

        Args:
            obs: batched observation tuple (from act_()).
            test: if True, use greedy (epsilon=0).

        Returns:
            np.ndarray: scalar action index.
        """
        if test:
            self._current_explore_count = 0
            self._last_explore_action = None

        if not test and self._current_explore_count > 0 and self._last_explore_action is not None:
            self._current_explore_count -= 1
            return self._last_explore_action

        with torch.no_grad():
            q_vals = self.forward(obs)  # (1, n_actions)

        if not test and np.random.random() < self.epsilon:
            action = np.array(np.random.randint(self.n_actions), dtype=np.int64)
            self._last_explore_action = action
            self._current_explore_count = self.explore_repeat_steps - 1
            return action

        self._current_explore_count = 0
        self._last_explore_action = None
        return q_vals.argmax(dim=-1).squeeze().cpu().numpy().astype(np.int64)
