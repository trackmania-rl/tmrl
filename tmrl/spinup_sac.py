# Adapted from the SAC implementation of OpenAI Spinup

from copy import deepcopy
from dataclasses import dataclass, InitVar
# from functools import lru_cache, reduce
# from torch.nn.functional import mse_loss

from tmrl.nn import no_grad, copy_shared
from tmrl.util import cached_property
# import tmrl.sac_models

import itertools
# import numpy as np
import torch
from torch.optim import Adam
# import gym
import time
import tmrl.spinup_sac_core as core


@dataclass(eq=0)
class SpinupSacAgent:  # Adapted from Spinup
    Env: InitVar

    actor_critic: type = core.MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    # lr = 1e-3
    alpha: float = 0.2
    Model: type = core.MLPActorCritic
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    # discount: float = 0.99  # reward discount factor
    # target_update: float = 0.005  # parameter for exponential moving average
    # reward_scale: float = 5.
    # entropy_scale: float = 1.
    device: str = None

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.Model(observation_space, action_space)
        print(f"DEBUG: device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.lr_actor)
        # self.critic_optimizer = torch.optim.Adam(self.model.critics.parameters(), lr=self.lr_critic)

        # logger.save_config(locals())

        # torch.manual_seed(seed)
        # np.random.seed(seed)

        # env, test_env = env_fn(), env_fn()
        # obs_dim = env.observation_space.shape
        # act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # act_limit = env.action_space.high[0]

        # Create actor-critic module and target networks
        # self.ac = self.actor_critic(env.observation_space, env.action_space)
        # self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        # for p in self.ac_targ.parameters():
        #     p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)

        # Experience buffer
        # replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # var_counts = tuple(core.count_vars(module) for module in [ac.actor, ac.q1, ac.q2])

        # logger.log('\nNumber of parameters: \t actor: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    def compute_loss_q(self, batch):
        o, a, r, o2, d = batch

        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.actor(o2)

            # Target Q-values
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging

        return loss_q

    # Set up function for computing SAC actor loss
    def compute_loss_pi(self, batch):
        o, _, _, _, _ = batch
        pi, logp_pi = self.model.actor(o)
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging

        return loss_pi

    def train(self, batch):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for actor.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        # logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return dict(
            loss_actor=loss_pi.detach(),
            loss_critic=loss_q.detach(),
        )
