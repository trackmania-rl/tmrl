"""SAC agent with hyperparameters from config (ALG_CONFIG)."""

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

import tmrl.config.constants as cfg
from tmrl.custom.custom_algorithms._common import (
    _amp_dtype,
    _amp_enabled,
    _compute_n_step_return_and_bootstrap_mask,
    _tensor_to_scalar,
    autocast_context,
    clip_model_weights,
    polyak_update,
    set_seed,
)
from tmrl.custom.models.mlp import MLPActorCritic
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.training import TrainingAgent
from tmrl.util import cached_property


@dataclass(eq=False)
class SpinupSacAgentConfig(TrainingAgent):
    """SAC agent with hyperparameters read from config (ALG_CONFIG).

    Same as SpinupSacAgent but lr_actor, lr_critic, lr_entropy, n_steps
    default to cfg.ALG_CONFIG values for config-driven training.
    """

    observation_space: type[Any]
    action_space: type[Any]
    device: str | None = None
    model_cls: type[Any] = MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = cfg.ALG_CONFIG["LR_ACTOR"]
    lr_critic: float = cfg.ALG_CONFIG["LR_CRITIC"]
    lr_entropy: float = cfg.ALG_CONFIG["LR_ENTROPY"]
    learn_entropy_coef: bool = True
    target_entropy: float | None = None
    n_steps: int = cfg.ALG_CONFIG["N_STEPS"]

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        set_seed()
        if self.n_steps == 1:
            self.n_steps = 0
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logger.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.actor_optimizer = Adam(
            self.model.actor.parameters(),
            lr=self.lr_actor,
            weight_decay=cfg.ACTOR_WEIGHT_DECAY,
            eps=cfg.ADAM_EPS,
        )
        self.critic_optimizer = Adam(
            itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
            lr=self.lr_critic,
            weight_decay=cfg.CRITIC_WEIGHT_DECAY,
            eps=cfg.ADAM_EPS,
        )
        self.use_mixed_precision = _amp_enabled(device)
        self.amp_dtype = _amp_dtype()
        use_scaler = self.use_mixed_precision and (self.amp_dtype != torch.bfloat16)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        if len(cfg.SCHEDULER_CONFIG["NAME"]) > 0:
            self.actor_scheduler = CosineAnnealingWarmRestarts(
                self.actor_optimizer,
                cfg.SCHEDULER_CONFIG["T_0"],
                cfg.SCHEDULER_CONFIG["T_mult"],
                cfg.SCHEDULER_CONFIG["eta_min"],
                cfg.SCHEDULER_CONFIG["last_epoch"],
            )

            self.critic_scheduler = CosineAnnealingWarmRestarts(
                self.critic_optimizer,
                cfg.SCHEDULER_CONFIG["T_0"],
                cfg.SCHEDULER_CONFIG["T_mult"],
                cfg.SCHEDULER_CONFIG["eta_min"],
                cfg.SCHEDULER_CONFIG["last_epoch"],
            )

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)
        if self.learn_entropy_coef:
            self.log_alpha = torch.log(
                torch.ones(1, device=self.device) * self.alpha
            ).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

        if cfg.WANDB_GRADIENTS and wandb is not None:
            wandb.watch(self.model, log_freq=10)

    def get_actor(self) -> Any:
        return self.model_nograd.actor

    def train(  # type: ignore[override]
        self, batch: tuple, epoch: int, batch_index: int, iters: int
    ) -> dict:
        if cfg.DEBUG_MODE:
            torch.autograd.set_detect_anomaly(True)
        o, a, r, o2, d = batch[0], batch[1], batch[2], batch[3], batch[4]

        def autocast_ctx():
            return autocast_context(self.use_mixed_precision, self.amp_dtype)

        batch_size = r.shape[0]
        if self.n_steps > 1 and self.n_steps >= batch_size:
            raise ValueError(
                f"Invalid n-step config: n_steps ({self.n_steps}) must be smaller than "
                f"batch_size ({batch_size})."
            )
        truncated_batch_size = batch_size if self.n_steps <= 1 else batch_size - self.n_steps

        with autocast_ctx():
            pi, logp_pi = self.model.actor(o)
        loss_alpha = None
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
        with autocast_ctx():
            q1 = self.model.q1(o, a)[:truncated_batch_size]
            q2 = self.model.q2(o, a)[:truncated_batch_size]

        n_step_not_done = None
        if self.n_steps > 1:
            n_step_return, n_step_not_done = _compute_n_step_return_and_bootstrap_mask(
                r, d, self.gamma, self.n_steps
            )
            r = n_step_return[:truncated_batch_size].squeeze(-1)
            # Do NOT slice o/o2: they are tuples; slicing would change tuple length, not batch dim.
            n_step_not_done = n_step_not_done[:truncated_batch_size].squeeze(-1)
        with torch.no_grad():
            with autocast_ctx():
                a2, logp_a2 = self.model.actor(o2)
            if self.n_steps > 1:
                logp_a2 = logp_a2[:truncated_batch_size]
                with autocast_ctx():
                    q1_pi_targ = self.model_target.q1(o2, a2)[:truncated_batch_size]
                    q2_pi_targ = self.model_target.q2(o2, a2)[:truncated_batch_size]
            else:
                with autocast_ctx():
                    q1_pi_targ = self.model_target.q1(o2, a2)
                    q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            alpha = alpha_t if alpha_t is not None else self.alpha_t
            assert alpha is not None
            if self.n_steps > 1:
                assert n_step_not_done is not None
                backup = r + (self.gamma**self.n_steps) * n_step_not_done * (
                    q_pi_targ.sub_(alpha * logp_a2)
                )
            else:
                backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
        with autocast_ctx():
            loss_q1 = (q1 - backup).pow(2).mean()
            loss_q2 = (q2 - backup).pow(2).mean()
            loss_critic = (loss_q1 + loss_q2) / 2

        self.critic_optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(loss_critic).backward()
            self.grad_scaler.step(self.critic_optimizer)
        else:
            loss_critic.backward()
            self.critic_optimizer.step()
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)
        if cfg.WEIGHT_CLIPPING_ENABLED:
            clip_model_weights(self.model.q1)
            clip_model_weights(self.model.q2)
        with autocast_ctx():
            q1_pi = self.model.q1(o, pi)[:truncated_batch_size]
            q2_pi = self.model.q2(o, pi)[:truncated_batch_size]
            q_pi = torch.min(q1_pi, q2_pi)
        with autocast_ctx():
            loss_actor = (alpha_t * logp_pi[:truncated_batch_size] - q_pi).mean()
        self.actor_optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(loss_actor).backward()
            self.grad_scaler.step(self.actor_optimizer)
            self.grad_scaler.update()
        else:
            loss_actor.backward()
            self.actor_optimizer.step()

        if len(cfg.SCHEDULER_CONFIG["NAME"]) > 0:
            self.actor_scheduler.step(epoch + batch_index / iters)
            self.critic_scheduler.step(epoch + batch_index / iters)
        if cfg.WEIGHT_CLIPPING_ENABLED:
            clip_model_weights(self.model.actor)
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)
        polyak_update(self.model, self.model_target, self.polyak)
        ret_dict = self._build_return_dict(
            loss_actor,
            loss_critic,
            loss_alpha,
            alpha_t,
            logp_pi,
            logp_a2,
            q1,
            q2,
            q_pi,
            q_pi_targ,
            q1_pi_targ,
            q2_pi_targ,
            q1_pi,
            q2_pi,
            backup,
            r,
            d,
            a,
            pi,
            a2,
            o,
            o2,
            truncated_batch_size,
        )
        return ret_dict

    def _build_return_dict(
        self,
        loss_actor,
        loss_critic,
        loss_alpha,
        alpha_t,
        logp_pi,
        logp_a2,
        q1,
        q2,
        q_pi,
        q_pi_targ,
        q1_pi_targ,
        q2_pi_targ,
        q1_pi,
        q2_pi,
        backup,
        r,
        d,
        a,
        pi,
        a2,
        o,
        o2,
        truncated_batch_size,
    ) -> dict:
        """Build the dict of scalars to log (and optionally debug metrics)."""
        with torch.no_grad():
            ret_dict = {
                "losses/actor": _tensor_to_scalar(loss_actor.detach()),
                "losses/critic": _tensor_to_scalar(loss_critic.detach()),
                "lrs/actor_lr": self.actor_optimizer.param_groups[0]["lr"],
                "lrs/critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            }
            if cfg.WANDB_DEBUG:
                ts = truncated_batch_size
                q1_o2_a2 = self.model.q1(o2, a2)[:ts]
                q2_o2_a2 = self.model.q2(o2, a2)[:ts]
                q1_targ_pi = self.model_target.q1(o, pi)[:ts]
                q2_targ_pi = self.model_target.q2(o, pi)[:ts]
                q1_targ_a = self.model_target.q1(o, a)[:ts]
                q2_targ_a = self.model_target.q2(o, a)[:ts]

                pairs = {
                    "debug/diff_q1pt_qpt": q1_pi_targ - q_pi_targ,
                    "debug/diff_q2pt_qpt": q2_pi_targ - q_pi_targ,
                    "debug/diff_q1_q1t_a2": q1_o2_a2 - q1_pi_targ,
                    "debug/diff_q2_q2t_a2": q2_o2_a2 - q2_pi_targ,
                    "debug/diff_q1_q1t_pi": q1_pi - q1_targ_pi,
                    "debug/diff_q2_q2t_pi": q2_pi - q2_targ_pi,
                    "debug/diff_q1_q1t_a": q1 - q1_targ_a,
                    "debug/diff_q2_q2t_a": q2 - q2_targ_a,
                    "debug/diff_q1": q1 - backup,
                    "debug/diff_q2": q2 - backup,
                    "debug/diff_r_q1": q1 - backup + r,
                    "debug/diff_r_q2": q2 - backup + r,
                }
                for key, val in pairs.items():
                    val = val.detach()
                    ret_dict[key] = _tensor_to_scalar(val.mean())
                    ret_dict[key + "_std"] = _tensor_to_scalar(val.std())

                scalars = {
                    "debug/log_pi": logp_pi,
                    "debug/logp_a2": logp_a2,
                    "debug/q_a1": q_pi,
                    "debug/q_a1_targ": q_pi_targ,
                    "debug/backup": backup,
                    "debug/q1": q1,
                    "debug/q2": q2,
                    "debug/r": r,
                    "debug/d": d,
                }
                for key, val in scalars.items():
                    val = val.detach()
                    ret_dict[key] = _tensor_to_scalar(val.mean())
                    ret_dict[key + "_std"] = _tensor_to_scalar(val.std())

                for label, tensor in [("a", a), ("a1", pi), ("a2", a2)]:
                    for dim in range(min(3, tensor.shape[-1])):
                        ret_dict[f"debug/{label}_{dim}"] = _tensor_to_scalar(
                            tensor[:, dim].detach().mean()
                        )
                        ret_dict[f"debug/{label}_{dim}_std"] = _tensor_to_scalar(
                            tensor[:, dim].detach().std()
                        )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict
