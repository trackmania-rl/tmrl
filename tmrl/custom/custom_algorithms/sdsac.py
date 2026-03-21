"""Stable Discrete SAC (SD-SAC) agent for discrete action spaces.

Implements the three stabilisation tricks from
"Revisiting Discrete Soft Actor-Critic" (Zhou et al., TMLR 2024):
  1. Double Average Q-learning  (--avg-q)
  2. Q-clip                     (--clip-q)
  3. Entropy Penalty             (--entropy-penalty)

References:
  - coldsummerday/SD-SAC (official implementation)
  - arXiv / OpenReview: https://openreview.net/forum?id=EUF2R6VBeU
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.distributions import Categorical
from torch.optim import Adam

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

import tmrl.config.constants as cfg
from tmrl.custom.custom_algorithms._common import (
    _compute_n_step_return_and_bootstrap_mask,
    _tensor_to_scalar,
    polyak_update,
    project_simbav2_weights,
    sanitize_obs,
    set_seed,
)
from tmrl.custom.models.DQNNet import DQNActor
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.custom.utils.optim import GradientStabilizer
from tmrl.training import TrainingAgent
from tmrl.util import cached_property, wandb_monotonic_step


class DiscreteQHead(nn.Module):
    """Simple Q-value head: maps features to per-action Q-values."""

    def __init__(self, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(features))


class DiscreteSACNetwork(nn.Module):
    """Actor-critic network for discrete SAC.

    The actor produces a Categorical distribution over actions.
    Each critic maps observation features to Q-values for every action.
    """

    def __init__(
        self,
        observation_space,
        n_actions: int,
        hidden_dim: int = 256,
        num_blocks_actor: int = 2,
        num_blocks_critic: int = 4,
        n_cos: int = 64,
    ):
        super().__init__()
        from tmrl.custom.models.DQNNet import IQNFeatureBackbone

        self.actor_backbone = IQNFeatureBackbone(
            observation_space, hidden_dim=hidden_dim, num_blocks=num_blocks_actor, n_cos=n_cos
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self.q1_backbone = IQNFeatureBackbone(
            observation_space, hidden_dim=hidden_dim, num_blocks=num_blocks_critic, n_cos=n_cos
        )
        self.q1_head = DiscreteQHead(hidden_dim, n_actions)

        self.q2_backbone = IQNFeatureBackbone(
            observation_space, hidden_dim=hidden_dim, num_blocks=num_blocks_critic, n_cos=n_cos
        )
        self.q2_head = DiscreteQHead(hidden_dim, n_actions)

        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

    def _dummy_tau(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Single quantile fraction = 0.5 to reuse IQNFeatureBackbone without IQN."""
        return torch.full((batch_size, 1), 0.5, device=device)

    def actor_logits(self, obs) -> torch.Tensor:
        """Return raw logits over actions.  Shape: (batch, n_actions)."""
        batch_size = obs[0].shape[0]
        tau = self._dummy_tau(batch_size, obs[0].device)
        features = self.actor_backbone(obs, tau).squeeze(1)
        return cast(torch.Tensor, self.actor_head(features))

    def q1(self, obs) -> torch.Tensor:
        batch_size = obs[0].shape[0]
        tau = self._dummy_tau(batch_size, obs[0].device)
        features = self.q1_backbone(obs, tau).squeeze(1)
        return cast(torch.Tensor, self.q1_head(features))

    def q2(self, obs) -> torch.Tensor:
        batch_size = obs[0].shape[0]
        tau = self._dummy_tau(batch_size, obs[0].device)
        features = self.q2_backbone(obs, tau).squeeze(1)
        return cast(torch.Tensor, self.q2_head(features))


class DiscreteSACActor(DQNActor):
    """Rollout actor for SD-SAC: samples from the softmax policy."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        n_cos: int = 64,
        n_actions: int = 78,
        epsilon: float = 0.01,
    ):
        super().__init__(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            n_cos=n_cos,
            dueling=False,
            n_actions=n_actions,
            epsilon=epsilon,
            n_quantiles_eval=1,
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def act(self, obs, test=False):
        with torch.no_grad():
            batch_size = obs[0].shape[0]
            tau = torch.full((batch_size, 1), 0.5, device=obs[0].device)
            features = self.q_net.backbone(obs, tau).squeeze(1)
            logits = self.actor_head(features)
            if test:
                return logits.argmax(dim=-1).squeeze().cpu().numpy().astype(np.int64)
            dist = Categorical(logits=logits)
            action = dist.sample()
        return action.squeeze().cpu().numpy().astype(np.int64)


@dataclass(eq=False)
class SDSACAgent(TrainingAgent):
    """Stable Discrete SAC agent with Double Avg Q, Q-clip, and Entropy Penalty.

    Designed for Trackmania with the same observation encoding as IQN but
    using a soft actor-critic framework for discrete actions.
    """

    observation_space: Any = None
    action_space: Any = None
    device: str | None = None

    # Architecture
    hidden_dim: int = 256
    num_blocks_actor: int = 2
    num_blocks_critic: int = 4
    n_cos: int = 64
    n_actions: int = 78

    # SAC hyper-parameters
    gamma: float = 0.99
    lr_actor: float = 5e-5
    lr_critic: float = 5e-5
    lr_alpha: float = 3e-5
    tau_polyak: float = 0.005
    n_steps: int = 3
    auto_alpha: bool = True
    alpha_init: float = 0.05

    # SD-SAC tricks
    use_avg_q: bool = True
    use_clip_q: bool = True
    clip_q_epsilon: float = 0.5
    use_entropy_penalty: bool = True
    entropy_penalty_beta: float = 0.5

    # Misc
    weight_decay: float = 0.0

    # EDER (set via config; 0 = disabled)
    eder_oversample_ratio: int = 0

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self) -> None:
        set_seed()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DiscreteSACNetwork(
            self.observation_space,
            n_actions=self.n_actions,
            hidden_dim=self.hidden_dim,
            num_blocks_actor=self.num_blocks_actor,
            num_blocks_critic=self.num_blocks_critic,
            n_cos=self.n_cos,
        ).to(device)

        self.model_target = no_grad(deepcopy(self.model))

        self.actor_optimizer = Adam(
            list(self.model.actor_backbone.parameters()) + list(self.model.actor_head.parameters()),
            lr=self.lr_actor,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = Adam(
            list(self.model.q1_backbone.parameters())
            + list(self.model.q1_head.parameters())
            + list(self.model.q2_backbone.parameters())
            + list(self.model.q2_head.parameters()),
            lr=self.lr_critic,
            weight_decay=self.weight_decay,
        )

        # Auto-tuned alpha
        self.log_alpha: torch.Tensor | None
        self.alpha_optimizer: Adam | None
        if self.auto_alpha:
            self.target_entropy = 0.98 * math.log(self.n_actions)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_alpha)
            self._alpha: float | torch.Tensor = self.log_alpha.detach().exp()
        else:
            self.target_entropy = 0.0
            self.log_alpha = None
            self.alpha_optimizer = None
            self._alpha = self.alpha_init

        self._grad_stabilizer_critic = GradientStabilizer(ema_decay=0.995)
        self._grad_stabilizer_actor = GradientStabilizer(ema_decay=0.995)
        self._training_step = 0
        logger.info(
            "SDSACAgent: n_actions={}, avg_q={}, clip_q={}, entropy_penalty={}",
            self.n_actions,
            self.use_avg_q,
            self.use_clip_q,
            self.use_entropy_penalty,
        )

    def get_actor(self) -> DiscreteSACActor:
        actor = DiscreteSACActor(
            self.observation_space,
            self.action_space,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks_actor,
            n_cos=self.n_cos,
            n_actions=self.n_actions,
            epsilon=0.0,
        )
        actor.q_net.backbone.load_state_dict(self.model.actor_backbone.state_dict())
        actor.actor_head.load_state_dict(self.model.actor_head.state_dict())
        return actor

    _sanitize_obs = staticmethod(sanitize_obs)

    def train(
        self,
        batch: tuple,
        epoch: int | None = None,
        batch_index: int | None = None,
        iters: int | None = None,
    ) -> dict[str, float]:
        self._training_step += 1

        o, a, r, o2, d = batch[0], batch[1], batch[2], batch[3], batch[4]
        o = self._sanitize_obs(o)
        o2 = self._sanitize_obs(o2)

        batch_size = r.shape[0]
        if self.n_steps > 1 and self.n_steps >= batch_size:
            raise ValueError(
                f"Invalid n-step config: n_steps ({self.n_steps}) must be smaller than "
                f"batch_size ({batch_size})."
            )
        actions = a.long().squeeze(-1)

        reward_scale = float(cfg.ALG_CONFIG.get("REWARD_NORMALIZE_SCALE", 1.0))
        if reward_scale != 1.0 and reward_scale > 0:
            r = r / reward_scale

        # -- EDER diversity filtering --
        if self.eder_oversample_ratio >= 2:
            from tmrl.custom.utils.eder import greedy_kdpp_filter

            with torch.no_grad():
                tau_dummy = torch.full((batch_size, 1), 0.5, device=o[0].device)
                feat = self.model.q1_backbone(o, tau_dummy).squeeze(1)
            target_k = batch_size // self.eder_oversample_ratio
            keep = greedy_kdpp_filter(feat, target_k)
            o = tuple(t[keep] for t in o)
            o2 = tuple(t[keep] for t in o2)
            a = a[keep]
            r = r[keep]
            d = d[keep]
            actions = actions[keep]
            batch_size = target_k

        # -- Sequence-aware n-step returns (ported from TQC) --
        burn_in_len = int(cfg.ALG_CONFIG.get("R2D2_BURN_IN", 0))
        seq_len = int(cfg.ALG_CONFIG.get("R2D2_SEQUENCE_LENGTH", 0))

        if self.n_steps > 1:
            truncated_batch_size = batch_size - self.n_steps
            n_step_return, bootstrap_mask = _compute_n_step_return_and_bootstrap_mask(
                r, d, self.gamma, self.n_steps
            )
            n_step_return = n_step_return[:truncated_batch_size].squeeze(-1)
            bootstrap_mask = bootstrap_mask[:truncated_batch_size].squeeze(-1)
            gamma_n = self.gamma**self.n_steps
            if seq_len > 0:
                step_in_seq = torch.arange(truncated_batch_size, device=r.device) % seq_len
                valid_n_step = (step_in_seq + self.n_steps <= seq_len) & (
                    step_in_seq >= burn_in_len
                )
            else:
                valid_n_step = None
        else:
            truncated_batch_size = batch_size
            if seq_len > 0:
                step_in_seq = torch.arange(truncated_batch_size, device=r.device) % seq_len
                valid_n_step = step_in_seq >= burn_in_len
            else:
                valid_n_step = None
            n_step_return = r[:truncated_batch_size].squeeze(-1)
            bootstrap_mask = (1.0 - d[:truncated_batch_size]).squeeze(-1)
            gamma_n = self.gamma

        o_t = tuple(t[:truncated_batch_size] for t in o)
        actions_t = actions[:truncated_batch_size]

        alpha_t = float(self._alpha) if isinstance(self._alpha, (int, float)) else self._alpha

        # -- Target Q --
        with torch.no_grad():
            if self.n_steps > 1:
                target_indices = (
                    torch.arange(truncated_batch_size, device=r.device) + self.n_steps - 1
                )
                o2_target = tuple(t[target_indices] for t in o2)
            else:
                o2_target = tuple(t[:truncated_batch_size] for t in o2)

            logits_next = self.model.actor_logits(o2_target)
            dist_next = Categorical(logits=logits_next)
            probs_next = dist_next.probs

            if self.use_avg_q:
                q_target = (self.model_target.q1(o2_target) + self.model_target.q2(o2_target)) * 0.5
            else:
                q_target = torch.min(
                    self.model_target.q1(o2_target), self.model_target.q2(o2_target)
                )

            v_next = (probs_next * q_target).sum(dim=-1) + alpha_t * dist_next.entropy()
            target = n_step_return + gamma_n * bootstrap_mask * v_next

        # -- Critic losses --
        current_q1_all = self.model.q1(o_t)
        current_q2_all = self.model.q2(o_t)
        current_q1 = current_q1_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        current_q2 = current_q2_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        if self.use_clip_q:
            with torch.no_grad():
                q1_old = self.model_target.q1(o_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                q2_old = self.model_target.q2(o_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            clipped_q1 = q1_old + (current_q1 - q1_old).clamp(
                -self.clip_q_epsilon, self.clip_q_epsilon
            )
            clipped_q2 = q2_old + (current_q2 - q2_old).clamp(
                -self.clip_q_epsilon, self.clip_q_epsilon
            )
            loss_q1 = torch.maximum((current_q1 - target) ** 2, (clipped_q1 - target) ** 2)
            loss_q2 = torch.maximum((current_q2 - target) ** 2, (clipped_q2 - target) ** 2)
            if valid_n_step is not None:
                denom = valid_n_step.float().sum().clamp(min=1.0)
                critic_loss = (loss_q1 * valid_n_step.float()).sum() / denom + (
                    loss_q2 * valid_n_step.float()
                ).sum() / denom
            else:
                critic_loss = loss_q1.mean() + loss_q2.mean()
        else:
            if valid_n_step is not None:
                denom = valid_n_step.float().sum().clamp(min=1.0)
                c1_loss = ((current_q1 - target) ** 2 * valid_n_step.float()).sum() / denom
                c2_loss = ((current_q2 - target) ** 2 * valid_n_step.float()).sum() / denom
                critic_loss = c1_loss + c2_loss
            else:
                critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._grad_stabilizer_critic.step(
            list(self.model.q1_backbone.parameters())
            + list(self.model.q1_head.parameters())
            + list(self.model.q2_backbone.parameters())
            + list(self.model.q2_head.parameters())
        )
        self.critic_optimizer.step()

        # -- Actor loss --
        logits = self.model.actor_logits(o_t)
        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        with torch.no_grad():
            if self.use_avg_q:
                q_for_actor = (self.model.q1(o_t) + self.model.q2(o_t)) * 0.5
            else:
                q_for_actor = torch.min(self.model.q1(o_t), self.model.q2(o_t))

        actor_loss_unmasked = -(alpha_t * entropy + (dist.probs * q_for_actor).sum(dim=-1))
        if valid_n_step is not None:
            denom = valid_n_step.float().sum().clamp(min=1.0)
            actor_loss = (actor_loss_unmasked * valid_n_step.float()).sum() / denom
        else:
            actor_loss = actor_loss_unmasked.mean()

        if self.use_entropy_penalty:
            with torch.no_grad():
                target_logits = self.model_target.actor_logits(o_t)
                target_dist = Categorical(logits=target_logits)
                target_entropy = target_dist.entropy()
            entropy_penalty = F.mse_loss(entropy, target_entropy)
            actor_loss = actor_loss + self.entropy_penalty_beta * entropy_penalty

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._grad_stabilizer_actor.step(
            list(self.model.actor_backbone.parameters()) + list(self.model.actor_head.parameters())
        )
        self.actor_optimizer.step()

        # -- Alpha update --
        alpha_loss_val = 0.0
        if self.auto_alpha and self.log_alpha is not None:
            log_prob = -entropy.detach() + self.target_entropy
            alpha_loss = -(self.log_alpha * log_prob).mean()
            assert self.alpha_optimizer is not None
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self._alpha = self.log_alpha.detach().exp()
            alpha_loss_val = float(alpha_loss.item())

        # -- Target network Polyak update --
        project_simbav2_weights(self.model)
        polyak_update(self.model, self.model_target, 1.0 - self.tau_polyak)

        # -- Logging --
        ret: dict[str, float] = {
            "loss/actor": _tensor_to_scalar(actor_loss),
            "loss/critic": _tensor_to_scalar(critic_loss),
            "state/entropy": _tensor_to_scalar(entropy.mean()),
            "state/q1": _tensor_to_scalar(current_q1.mean()),
            "state/q2": _tensor_to_scalar(current_q2.mean()),
            "state/q_target": _tensor_to_scalar(target.mean()),
            "debug/critic_grad_norm": critic_grad_norm,
            "debug/actor_grad_norm": actor_grad_norm,
            "debug/critic_grad_ema": self._grad_stabilizer_critic.ema_norm,
            "debug/actor_grad_ema": self._grad_stabilizer_actor.ema_norm,
            "alpha": (
                float(self._alpha)
                if isinstance(self._alpha, (int, float))
                else float(self._alpha.item())
            ),
            "train/step": self._training_step,
        }
        if self.auto_alpha:
            ret["loss/alpha"] = alpha_loss_val

        if wandb is not None and wandb.run is not None:
            wandb.log(ret, step=wandb_monotonic_step(self._training_step, wandb.run))

        return ret
