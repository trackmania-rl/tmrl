"""REDQ-SAC (Randomized Ensemble Double Q-learning SAC) agent."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.optim import Adam

from tmrl.custom.custom_algorithms._common import (
    _amp_dtype,
    _amp_enabled,
    autocast_context,
    polyak_update,
    set_seed,
)
from tmrl.custom.models.mlp import REDQMLPActorCritic
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.training import TrainingAgent
from tmrl.util import cached_property


@dataclass(eq=False)
class REDQSACAgent(TrainingAgent):
    """REDQ (Randomized Ensemble Double Q-learning) SAC agent.

    Uses an ensemble of Q-networks; each update samples a subset for the target
    to reduce overestimation while allowing more gradient updates per environment
    step (UTD ratio > 1).
    """

    observation_space: type[Any]
    action_space: type[Any]
    device: str | None = None
    model_cls: type[Any] = REDQMLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_entropy: float = 1e-3
    learn_entropy_coef: bool = True
    target_entropy: float | None = None
    n: int = 10
    m: int = 2
    q_updates_per_policy_update: int = 1

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        set_seed()

        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logger.debug(f" device REDQ-SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.pi_optimizer = Adam(
            self.model.actor.parameters(), lr=self.lr_actor, weight_decay=0.001
        )
        self.q_optimizer_list = [
            Adam(q.parameters(), lr=self.lr_critic, weight_decay=0.001) for q in self.model.qs
        ]
        self.criterion = torch.nn.MSELoss()
        self.use_mixed_precision = _amp_enabled(device)
        self.amp_dtype = _amp_dtype()
        use_scaler = self.use_mixed_precision and (self.amp_dtype != torch.bfloat16)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.i_update = 0
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

    def get_actor(self) -> Any:
        return self.model_nograd.actor

    def train(  # type: ignore[override]
        self, batch: tuple, epoch: int, batch_index: int, iters: int
    ) -> dict[str, float]:
        self.i_update += 1
        update_policy = self.i_update % self.q_updates_per_policy_update == 0

        o, a, r, o2, d, _ = batch
        pi, logp_pi = None, None

        def autocast_ctx():
            return autocast_context(self.use_mixed_precision, self.amp_dtype)

        if update_policy:
            with autocast_ctx():
                pi, logp_pi = self.model.actor(o)
        loss_alpha = None
        if self.learn_entropy_coef and update_policy and logp_pi is not None:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        with torch.no_grad(), autocast_ctx():
            a2, logp_a2 = self.model.actor(o2)

            sample_idxs = np.random.choice(self.n, self.m, replace=False)

            q_prediction_next_list = [self.model_target.qs[i](o2, a2) for i in sample_idxs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (
                min_q - alpha_t * logp_a2.unsqueeze(dim=-1)
            )

        with autocast_ctx():
            q_prediction_list = [q(o, a) for q in self.model.qs]
            q_prediction_cat = torch.stack(q_prediction_list, -1)
            backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

            loss_q = self.criterion(q_prediction_cat, backup)

        for q in self.q_optimizer_list:
            q.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(loss_q).backward()
        else:
            loss_q.backward()

        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            with autocast_ctx():
                qs_pi = [q(o, pi) for q in self.model.qs]
                qs_pi_cat = torch.stack(qs_pi, -1)
                ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
                assert logp_pi is not None
                loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()
            self.pi_optimizer.zero_grad()
            if self.use_mixed_precision:
                self.grad_scaler.scale(loss_pi).backward()
            else:
                loss_pi.backward()

            for q in self.model.qs:
                q.requires_grad_(True)

        for q_optimizer in self.q_optimizer_list:
            if self.use_mixed_precision:
                self.grad_scaler.step(q_optimizer)
            else:
                q_optimizer.step()

        if update_policy:
            if self.use_mixed_precision:
                self.grad_scaler.step(self.pi_optimizer)
            else:
                self.pi_optimizer.step()
        if self.use_mixed_precision:
            self.grad_scaler.update()

        polyak_update(self.model, self.model_target, self.polyak)

        ret_dict: dict[str, float] = {"losses/critic": loss_q.detach().item()}
        if update_policy:
            ret_dict["losses/actor"] = loss_pi.detach().item()

        if self.learn_entropy_coef and update_policy and loss_alpha is not None:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict
