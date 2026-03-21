"""Soft Actor-Critic (SAC) agent."""

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.optim import SGD, Adam, AdamW

import tmrl.config.constants as cfg
from tmrl.custom.models.mlp import MLPActorCritic
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.training import TrainingAgent
from tmrl.util import cached_property


@dataclass(eq=False)
class SpinupSacAgent(TrainingAgent):
    """Soft Actor-Critic (SAC) agent with optional learnable entropy coefficient.

    Adapted from Spinning Up in Deep RL. Supports SAC v1 (fixed alpha) and
    SAC v2 (learnable alpha with target entropy). Uses two Q-networks and
    min-Q target for value estimation.
    """

    observation_space: type[Any]
    action_space: type[Any]
    device: str | None = None
    model_cls: type[Any] = MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_entropy: float = 1e-3
    learn_entropy_coef: bool = True
    target_entropy: float | None = None
    optimizer_actor: str = "adam"
    optimizer_critic: str = "adam"
    betas_actor: tuple[float, ...] | None = None
    betas_critic: tuple[float, ...] | None = None
    l2_actor: float | None = None
    l2_critic: float | None = None

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self) -> None:
        """Build model, target, optimizers, and entropy coefficient (if learned)."""
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logger.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.optimizer_actor = self.optimizer_actor.lower()
        self.optimizer_critic = self.optimizer_critic.lower()
        if self.optimizer_actor not in ["adam", "adamw", "sgd"]:
            logger.warning(
                f"actor optimizer {self.optimizer_actor} is not valid, defaulting to sgd"
            )
        if self.optimizer_critic not in ["adam", "adamw", "sgd"]:
            logger.warning(
                f"critic optimizer {self.optimizer_critic} is not valid, defaulting to sgd"
            )
        pi_optimizer_cls: type[Adam] | type[AdamW] | type[SGD]
        if self.optimizer_actor == "adam":
            pi_optimizer_cls = Adam
        elif self.optimizer_actor == "adamw":
            pi_optimizer_cls = AdamW
        else:
            pi_optimizer_cls = SGD
        pi_optimizer_kwargs: dict[str, Any] = {"lr": self.lr_actor}
        if self.optimizer_actor in ["adam", "adamw"] and self.betas_actor is not None:
            pi_optimizer_kwargs["betas"] = tuple(self.betas_actor)
        if self.l2_actor is not None:
            pi_optimizer_kwargs["weight_decay"] = self.l2_actor

        q_optimizer_cls: type[Adam] | type[AdamW] | type[SGD]
        if self.optimizer_critic == "adam":
            q_optimizer_cls = Adam
        elif self.optimizer_critic == "adamw":
            q_optimizer_cls = AdamW
        else:
            q_optimizer_cls = SGD
        q_optimizer_kwargs: dict[str, Any] = {"lr": self.lr_critic}
        if self.optimizer_critic in ["adam", "adamw"] and self.betas_critic is not None:
            q_optimizer_kwargs["betas"] = tuple(self.betas_critic)
        if self.l2_critic is not None:
            q_optimizer_kwargs["weight_decay"] = self.l2_critic

        self.pi_optimizer = pi_optimizer_cls(self.model.actor.parameters(), **pi_optimizer_kwargs)
        self.q_optimizer = q_optimizer_cls(
            itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
            **q_optimizer_kwargs,
        )
        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(
                torch.ones(1, device=self.device) * self.alpha
            ).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        """Return the current actor (policy) module for rollout workers."""
        return self.model_nograd.actor

    def train(self, batch, epoch=None, batch_index=None, iters=None):
        """Perform one SAC training step on the given batch.

        Args:
            batch: Tuple (obs, actions, rewards, next_obs, dones, info).
            epoch: Current epoch (unused, for API compat with training loop).
            batch_index: Current batch index (unused).
            iters: Total iterations (unused).

        Returns:
            Dict with losses/actor, losses/critic, and optionally loss_entropy_coef,
            entropy_coef and debug metrics when DEBUG_MODE is True.
        """
        obs, actions, rewards, next_obs, dones, _ = batch

        policy_actions, log_prob_pi = self.model.actor(obs)
        alpha_t, loss_alpha = self._sac_update_entropy_coef(policy_actions, log_prob_pi)
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        q1_pred = self.model.q1(obs, actions)
        q2_pred = self.model.q2(obs, actions)
        td_target = self._sac_compute_td_target(next_obs, rewards, dones, alpha_t)
        loss_q1 = ((q1_pred - td_target) ** 2).mean()
        loss_q2 = ((q2_pred - td_target) ** 2).mean()
        loss_q = (loss_q1 + loss_q2) / 2

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        q1_pi = self.model.q1(obs, policy_actions)
        q2_pi = self.model.q2(obs, policy_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha_t * log_prob_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        self._sac_update_target_network()
        ret_dict = self._sac_build_return_dict(
            loss_pi,
            loss_q,
            alpha_t,
            loss_alpha,
            obs,
            actions,
            next_obs,
            dones,
            rewards,
            policy_actions,
            log_prob_pi,
            q1_pred,
            q2_pred,
            q1_pi,
            q2_pi,
            q_pi,
            td_target,
        )
        return ret_dict

    def _sac_update_entropy_coef(self, policy_actions, log_prob_pi):
        """Compute current alpha and optional entropy loss for SAC v2.

        Args:
            policy_actions: Unused; kept for signature compatibility.
            log_prob_pi: Log probability of policy actions.

        Returns:
            Tuple (alpha_t, loss_alpha or None).
        """
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
            return alpha_t, loss_alpha
        return self.alpha_t, None

    def _sac_compute_td_target(self, next_obs, rewards, dones, alpha_t):
        """Compute Bellman backup (TD target) for Q-learning."""
        with torch.no_grad():
            next_actions, log_prob_next = self.model.actor(next_obs)
            q1_next = self.model_target.q1(next_obs, next_actions)
            q2_next = self.model_target.q2(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            return rewards + self.gamma * (1 - dones) * (min_q_next - alpha_t * log_prob_next)

    def _sac_update_target_network(self) -> None:
        """Polyak-update target network parameters."""
        with torch.no_grad():
            for param, param_targ in zip(
                self.model.parameters(), self.model_target.parameters(), strict=True
            ):
                param_targ.data.mul_(self.polyak)
                param_targ.data.add_((1 - self.polyak) * param.data)

    def _sac_build_return_dict(
        self,
        loss_pi,
        loss_q,
        alpha_t,
        loss_alpha,
        obs,
        actions,
        next_obs,
        dones,
        rewards,
        policy_actions,
        log_prob_pi,
        q1_pred,
        q2_pred,
        q1_pi,
        q2_pi,
        q_pi,
        td_target,
    ):
        """Build the dict of scalars to log (and optionally debug metrics)."""
        with torch.no_grad():
            if not cfg.DEBUG_MODE:
                ret_dict = {
                    "losses/actor": loss_pi.detach().item(),
                    "losses/critic": loss_q.detach().item(),
                }
            else:
                next_actions, log_prob_next = self.model.actor(next_obs)
                q1_next_obs_next_a = self.model.q1(next_obs, next_actions)
                q2_next_obs_next_a = self.model.q2(next_obs, next_actions)
                q1_targ_pi = self.model_target.q1(obs, policy_actions)
                q2_targ_pi = self.model_target.q2(obs, policy_actions)
                q1_targ_a = self.model_target.q1(obs, actions)
                q2_targ_a = self.model_target.q2(obs, actions)
                q1_pi_targ = self.model_target.q1(next_obs, next_actions)
                q2_pi_targ = self.model_target.q2(next_obs, next_actions)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

                ret_dict = {
                    "losses/actor": loss_pi.detach().item(),
                    "losses/critic": loss_q.detach().item(),
                    "debug_log_pi": log_prob_pi.detach().mean().item(),
                    "debug_log_pi_std": log_prob_pi.detach().std().item(),
                    "debug_logp_a2": log_prob_next.detach().mean().item(),
                    "debug_logp_a2_std": log_prob_next.detach().std().item(),
                    "debug_q_a1": q_pi.detach().mean().item(),
                    "debug_q_a1_std": q_pi.detach().std().item(),
                    "debug_q_a1_targ": q_pi_targ.detach().mean().item(),
                    "debug_q_a1_targ_std": q_pi_targ.detach().std().item(),
                    "debug_backup": td_target.detach().mean().item(),
                    "debug_backup_std": td_target.detach().std().item(),
                    "debug_q1": q1_pred.detach().mean().item(),
                    "debug_q1_std": q1_pred.detach().std().item(),
                    "debug_q2": q2_pred.detach().mean().item(),
                    "debug_q2_std": q2_pred.detach().std().item(),
                    "debug_diff_q1": (q1_pred - td_target).detach().mean().item(),
                    "debug_diff_q1_std": (q1_pred - td_target).detach().std().item(),
                    "debug_diff_q2": (q2_pred - td_target).detach().mean().item(),
                    "debug_diff_q2_std": (q2_pred - td_target).detach().std().item(),
                    "debug_diff_r_q1": (q1_pred - td_target + rewards).detach().mean().item(),
                    "debug_diff_r_q1_std": (q1_pred - td_target + rewards).detach().std().item(),
                    "debug_diff_r_q2": (q2_pred - td_target + rewards).detach().mean().item(),
                    "debug_diff_r_q2_std": (q2_pred - td_target + rewards).detach().std().item(),
                    "debug_diff_q1pt_qpt": (q1_pi_targ - q_pi_targ).detach().mean().item(),
                    "debug_diff_q2pt_qpt": (q2_pi_targ - q_pi_targ).detach().mean().item(),
                    "debug_diff_q1_q1t_a2": (q1_next_obs_next_a - q1_pi_targ)
                    .detach()
                    .mean()
                    .item(),
                    "debug_diff_q2_q2t_a2": (q2_next_obs_next_a - q2_pi_targ)
                    .detach()
                    .mean()
                    .item(),
                    "debug_diff_q1_q1t_pi": (q1_pi - q1_targ_pi).detach().mean().item(),
                    "debug_diff_q2_q2t_pi": (q2_pi - q2_targ_pi).detach().mean().item(),
                    "debug_diff_q1_q1t_a": (q1_pred - q1_targ_a).detach().mean().item(),
                    "debug_diff_q2_q2t_a": (q2_pred - q2_targ_a).detach().mean().item(),
                    "debug_diff_q1pt_qpt_std": (q1_pi_targ - q_pi_targ).detach().std().item(),
                    "debug_diff_q2pt_qpt_std": (q2_pi_targ - q_pi_targ).detach().std().item(),
                    "debug_diff_q1_q1t_a2_std": (q1_next_obs_next_a - q1_pi_targ)
                    .detach()
                    .std()
                    .item(),
                    "debug_diff_q2_q2t_a2_std": (q2_next_obs_next_a - q2_pi_targ)
                    .detach()
                    .std()
                    .item(),
                    "debug_diff_q1_q1t_pi_std": (q1_pi - q1_targ_pi).detach().std().item(),
                    "debug_diff_q2_q2t_pi_std": (q2_pi - q2_targ_pi).detach().std().item(),
                    "debug_diff_q1_q1t_a_std": (q1_pred - q1_targ_a).detach().std().item(),
                    "debug_diff_q2_q2t_a_std": (q2_pred - q2_targ_a).detach().std().item(),
                    "debug_r": rewards.detach().mean().item(),
                    "debug_r_std": rewards.detach().std().item(),
                    "debug_d": dones.detach().mean().item(),
                    "debug_d_std": dones.detach().std().item(),
                    "debug_a_0": actions[:, 0].detach().mean().item(),
                    "debug_a_0_std": actions[:, 0].detach().std().item(),
                    "debug_a_1": actions[:, 1].detach().mean().item(),
                    "debug_a_1_std": actions[:, 1].detach().std().item(),
                    "debug_a_2": actions[:, 2].detach().mean().item(),
                    "debug_a_2_std": actions[:, 2].detach().std().item(),
                    "debug_a1_0": policy_actions[:, 0].detach().mean().item(),
                    "debug_a1_0_std": policy_actions[:, 0].detach().std().item(),
                    "debug_a1_1": policy_actions[:, 1].detach().mean().item(),
                    "debug_a1_1_std": policy_actions[:, 1].detach().std().item(),
                    "debug_a1_2": policy_actions[:, 2].detach().mean().item(),
                    "debug_a1_2_std": policy_actions[:, 2].detach().std().item(),
                    "debug_a2_0": next_actions[:, 0].detach().mean().item(),
                    "debug_a2_0_std": next_actions[:, 0].detach().std().item(),
                    "debug_a2_1": next_actions[:, 1].detach().mean().item(),
                    "debug_a2_1_std": next_actions[:, 1].detach().std().item(),
                    "debug_a2_2": next_actions[:, 2].detach().mean().item(),
                    "debug_a2_2_std": next_actions[:, 2].detach().std().item(),
                }

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict
