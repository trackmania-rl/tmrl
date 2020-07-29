from collections import deque
from copy import deepcopy, copy
from dataclasses import dataclass, InitVar
from functools import lru_cache, reduce
from itertools import chain
import numpy as np
import torch
from torch.nn.functional import mse_loss

from agents.memory import TrajMemoryNoHidden
from agents.nn import PopArt, no_grad, copy_shared, exponential_moving_average, hd_conv
from agents.util import cached_property, partial

from agents.sac_models_rd import Mlp
from agents.envs import RandomDelayEnv


def print_debug(st):
    return
    print("DEBUG: " + st)


@dataclass(eq=0)
class Agent:
    Env: InitVar

    Model: type = Mlp
    OutputNorm: type = PopArt
    batchsize: int = 256  # training batch size
    memory_size: int = 1000000  # replay memory size
    lr: float = 0.0003  # learning rate
    discount: float = 0.99  # reward discount factor
    target_update: float = 0.005  # parameter for exponential moving average
    reward_scale: float = 5.
    entropy_scale: float = 1.
    start_training: int = 10000
    device: str = None
    training_steps: float = 1.  # training steps per environment interaction step

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    total_updates = 0  # will be (len(self.memory)-start_training) * training_steps / training_interval
    environment_steps = 0

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space
            self.sup_obs_delay = env.obs_delay_range.stop
            self.sup_act_delay = env.act_delay_range.stop
            # # print_debug(f"self.sup_obs_delay: {self.sup_obs_delay}")
            # # print_debug(f"self.sup_act_delay: {self.sup_act_delay}")
            self.act_buf_size = self.sup_obs_delay + self.sup_act_delay - 1

        assert self.device is not None
        device = self.device
        model = self.Model(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critics.parameters(), lr=self.lr)
        self.memory = TrajMemoryNoHidden(self.memory_size, self.batchsize, device, history=self.act_buf_size + 1)  # + 1 because we need the reward that comes after the last augmented state

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

    def act(self, state, obs, r, done, info, train=False):
        stats = []
        state = self.model.reset() if state is None else state  # initialize state if necessary
        action, next_state, _ = self.model.act(state, obs, r, done, info, train)

        if train:
            self.memory.append(np.float32(r), np.float32(done), info, obs, action)
            self.environment_steps += 1

            total_updates_target = (self.environment_steps - self.start_training) * self.training_steps
            while self.total_updates < int(total_updates_target):
                if self.total_updates == 0:
                    print("starting training")
                stats += self.train(),
                self.total_updates += 1
        return action, next_state, stats

    def train(self):
        augm_obs_traj, act_traj, rew_traj, terminals = self.memory.sample()
        batch_size = terminals.shape[0]
        # print_debug(f"batch_size: {batch_size}")
        # print_debug(f"augm_obs_traj: {augm_obs_traj}")
        # print_debug(f"act_traj: {act_traj}")
        # print_debug(f"rew_traj: {rew_traj}")
        # print_debug(f"terminals: {terminals}")

        obs = augm_obs_traj[0]
        next_obs = augm_obs_traj[1]
        actions = act_traj[0]  # FIXME: are these the actions we are looking for?
        rewards = rew_traj[0] * 0.0
        # print_debug(f"obs: {obs}")
        # print_debug(f"next_obs: {next_obs}")
        # print_debug(f"actions: {actions}")
        # print_debug(f"rewards: {rewards}")

        # We are looking for all the time steps at which the first action was applied
        # to determine the length of the n-step backup, nstep_len_min is the time at which the currently computed action (== i) or any action that followed (< i) has been applied first:
        # when nstep_len_min is k (in 0..self.act_buf_size-1), it means that the action computed with the first augmented observation of the trajectory will have an effect k+1 steps later
        # (either it will be applied, or an action that follows it will be applied)
        int_tens_type = obs_del = augm_obs_traj[0][2].dtype
        ones_tens = torch.ones(batch_size, device=self.device, dtype=int_tens_type, requires_grad=False)
        nstep_len_min = ones_tens * self.act_buf_size
        nstep_len_max = ones_tens * -1
        for i in reversed(range(self.act_buf_size)):  # caution: we don't care about the delay of the first observation in the trajectory, but we care about the last one
            obs_del = augm_obs_traj[i + 1][2]
            act_del = augm_obs_traj[i + 1][3]
            tot_del = obs_del + act_del
            # print_debug(f"i + 1: {i + 1}")
            # print_debug(f"obs_del: {obs_del}")
            # print_debug(f"act_del: {act_del}")
            # print_debug(f"tot_del: {tot_del}")
            # print_debug(f"nstep_len_min before: {nstep_len_min}")
            # print_debug(f"nstep_len_max before: {nstep_len_max}")
            nstep_len_min = torch.where((tot_del == i) & (tot_del < nstep_len_min), ones_tens * i, nstep_len_min)
            nstep_len_max = torch.where((tot_del == i) & (tot_del > nstep_len_max), ones_tens * i, nstep_len_max)
            # print_debug(f"nstep_len_min after: {nstep_len_min}")
            # print_debug(f"nstep_len_max after: {nstep_len_max}")
        # print_debug(f"nstep_len_min: {nstep_len_min}")
        # print_debug(f"nstep_len_max: {nstep_len_max}")
        # nstep_max_len = torch.max(nstep_len_min)
        # assert nstep_max_len < self.act_buf_size, "Delays longer than the action buffer (e.g. infinite) are not supported"
        # # print_debug(f"nstep_max_len: {nstep_max_len}")
        # nstep_one_hot = torch.zeros(len(nstep_len_min), nstep_max_len + 1, device=self.device, requires_grad=False).scatter_(1, nstep_len_min.unsqueeze(1), 1.)

        # Now we modify the rewards in the following way:
        # When the computed action will not be applied -> r = 0
        # Else -> r = discounted sum of all the rewards directly generated by this action

        for i in range(self.act_buf_size):
            # print_debug(f"i: {i}")
            # print_debug(f"rewards before: {rewards}")
            # print_debug(f"where: {torch.where((nstep_len_min <= i) & (nstep_len_max >= i), rewards + np.power(self.discount, i) * rew_traj[i + 1], rewards)}")
            # FIXME: maybe we want to discount by i instead of i+1?
            rewards = torch.where((nstep_len_min <= i) & (nstep_len_max >= i), rewards + np.power(self.discount, i + 1) * rew_traj[i + 1], rewards)  # rew_traj[i + 1] because, when tot_del is 0 (rtrl setting), the action has an effect only on the reward at index 1 in rew_traj
            # print_debug(f"rewards after: {rewards}")

        new_action_distribution = self.model.actor(obs)  # outputs distribution object
        new_actions = new_action_distribution.rsample()  # samples using the reparametrization trick

        # critic loss
        next_action_distribution = self.model_nograd.actor(next_obs)  # outputs distribution object
        next_actions = next_action_distribution.sample()  # samples
        next_value = [c(next_obs, next_actions) for c in self.model_target.critics]
        next_value = reduce(torch.min, next_value)  # minimum action-value
        next_value = self.outputnorm_target.unnormalize(next_value)  # PopArt (not present in the original paper)
        # next_value = self.outputnorm.unnormalize(next_value)  # PopArt (not present in the original paper)

        # predict entropy rewards in a separate dimension from the normal rewards (not present in the original paper)
        next_action_entropy = - (1. - terminals) * self.discount * next_action_distribution.log_prob(next_actions)
        reward_components = torch.cat((
            self.reward_scale * rewards[:, None],
            self.entropy_scale * next_action_entropy[:, None],
        ), dim=1)  # shape = (batchsize, reward_components)

        value_target = reward_components + (1. - terminals[:, None]) * self.discount * next_value
        normalized_value_target = self.outputnorm.update(value_target)  # PopArt update and normalize

        values = [c(obs, actions) for c in self.model.critics]
        assert values[0].shape == normalized_value_target.shape and not normalized_value_target.requires_grad
        loss_critic = sum(mse_loss(v, normalized_value_target) for v in values)

        # update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # actor loss
        new_value = [c(obs, new_actions) for c in self.model.critics]  # new_actions with reparametrization trick
        new_value = reduce(torch.min, new_value)  # minimum action_values
        assert new_value.shape == (self.batchsize, 2)

        new_value = self.outputnorm.unnormalize(new_value)
        new_value[:, -1] -= self.entropy_scale * new_action_distribution.log_prob(new_actions)
        loss_actor = - self.outputnorm.normalize_sum(new_value.sum(1)).mean()  # normalize_sum preserves relative scale

        # update actor
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # update target critics and normalizers
        exponential_moving_average(self.model_target.critics.parameters(), self.model.critics.parameters(), self.target_update)
        exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)

        return dict(
            loss_actor=loss_actor.detach(),
            loss_critic=loss_critic.detach(),
            outputnorm_reward_mean=self.outputnorm.mean[0],
            outputnorm_entropy_mean=self.outputnorm.mean[-1],
            outputnorm_reward_std=self.outputnorm.std[0],
            outputnorm_entropy_std=self.outputnorm.std[-1],
            memory_size=len(self.memory),
        )


# === tests ============================================================================================================
def test_agent():
    from agents import Training, run
    Delayed_Sac_Test = partial(
        Training,
        epochs=2,
        rounds=10,
        Agent=partial(Agent, start_training=256, batchsize=256, device='cuda', Model=partial(Mlp, act_delay=True, obs_delay=True)),
        Env=partial(RandomDelayEnv, min_observation_delay=0, sup_observation_delay=2, min_action_delay=0, sup_action_delay=2),  # RTRL setting, should get roughly the same behavior as SAC in RTRL
    )
    # Sac_Test = partial(
    #     Training,
    #     epochs=3,
    #     rounds=5,
    #     steps=100,
    #     Agent=partial(Agent, device='cpu', memory_size=1000000, start_training=256, batchsize=4),
    #     Env=partial(id="Pendulum-v0", real_time=0),
    # )
    run(Delayed_Sac_Test)


if __name__ == "__main__":
    test_agent()
# test_agent_avenue()
# test_agent_avenue_hd()
