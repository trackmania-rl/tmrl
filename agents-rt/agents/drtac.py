# Delayed RTAC

from copy import deepcopy
from dataclasses import dataclass
from functools import reduce

import pandas
import torch
from torch.nn.functional import mse_loss

import agents.sac
import agents.sac_undelayed
from agents.memory import TrajMemoryNoHidden
from agents.nn import no_grad, exponential_moving_average
from agents.util import partial

from agents.drtac_models import Mlp
from agents.envs import RandomDelayEnv

from agents import Training


def print_debug(st):
    # return
    print("DEBUG: " + st)


class DcacInterface:
    def replace_act_buf_in_augm_obs_tensor(self, augm_obs_tensor, act_buf_tensor):
        """
        must return a tensor with replaced action buffer
        """

        # return mod_augm_obs_tensor

        raise NotImplementedError

    def get_act_buf_in_augm_obs_tensor(self, augm_obs_tensor, act_buf_tensor):
        """
        must return a tensor with replaced action buffer
        """

        # return mod_augm_obs_tensor

        raise NotImplementedError


@dataclass(eq=0)
class Agent(agents.sac.Agent):
    Model: type = Mlp
    loss_alpha: float = 0.2
    rtac: bool = False
    constant: int = 0  # if > 0, no delays are looked up in the observation, instead the total delay is set to constant (caution: this is the true constant delays, including the inference delay, contrary to the action delay used in the rest of the algorithm)

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space
            if self.constant:
                self.act_buf_size = self.constant
            else:
                self.sup_obs_delay = env.obs_delay_range.stop
                self.sup_act_delay = env.act_delay_range.stop
                self.act_buf_size = self.sup_obs_delay + self.sup_act_delay - 1  # - 1 because self.sup_act_delay is actually max_act_delay as defined in the paper (self.sup_obs_delay is max_obs_delay+1)
            # print_debug(f"self.sup_obs_delay: {self.sup_obs_delay}")
            # print_debug(f"self.sup_act_delay: {self.sup_act_delay}")
            self.old_act_buf_size = deepcopy(self.act_buf_size)
            if self.rtac:
                # print_debug(f"RTAC FLAG IS TRUE, HISTORY OF SIZE 1")
                self.act_buf_size = 1

        assert self.device is not None
        device = self.device  # or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.Model(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = self.Memory(self.memory_size, self.batchsize, device)
        self.traj_new_actions = [None, ] * self.act_buf_size
        self.traj_new_actions_detach = [None, ] * self.act_buf_size
        self.traj_new_actions_log_prob = [None, ] * self.act_buf_size
        self.traj_new_actions_log_prob_detach = [None, ] * self.act_buf_size
        self.traj_new_augm_obs = [None, ] * (self.act_buf_size + 1)  # + 1 because the trajectory is obs0 -> rew1(obs0,act0) -> obs1 -> ...

        self.is_training = False

    def train(self):
        # TODO: remove requires_grad everywhere it should not be

        # sample a trajectory of length self.act_buf_size
        augm_obs_traj, rew_traj, done_traj = self.memory.sample()
        # TODO: act_traj is useless, it could be removed from the replay memory

        batch_size = done_traj.shape[0]
        print_debug(f"batch_size: {batch_size}")

        print_debug(f"augm_obs_traj: {augm_obs_traj}")
        print_debug(f"rew_traj: {rew_traj}")
        print_debug(f"done_traj: {done_traj}")

        print("DEBUG EXIT")
        exit()

        # value of the first augmented state:
        values = [c(augm_obs_traj[0]).squeeze() for c in self.model.critics]
        # print_debug(f"values: {values}")

        # to determine the length of the n-step backup, nstep_len is the time at which the currently computed action (== i) or any action that followed (< i) has been applied first:
        # when nstep_len is k (in 0..self.act_buf_size-1), it means that the action computed with the first augmented observation of the trajectory will have an effect k+1 steps later
        # (either it will be applied, or an action that follows it will be applied)
        int_tens_type = obs_del = augm_obs_traj[0][2].dtype
        ones_tens = torch.ones(batch_size, device=self.device, dtype=int_tens_type, requires_grad=False)
        zeros_tens = torch.zeros(batch_size, device=self.device, dtype=int_tens_type, requires_grad=False)

        if not self.rtac:
            nstep_len = ones_tens * self.act_buf_size
            for i in reversed(range(self.act_buf_size)):  # caution: we don't care about the delay of the first observation in the trajectory, but we care about the last one
                if self.constant:
                    tot_del = ones_tens * (self.constant - 1)  # -1 because of the convention we use for action delays
                else:
                    obs_del = augm_obs_traj[i + 1][2]
                    act_del = augm_obs_traj[i + 1][3]
                    tot_del = obs_del + act_del
                done = done_traj[i + 1]
                # print_debug(f"i + 1: {i + 1}")
                # print_debug(f"obs_del: {obs_del}")
                # print_debug(f"act_del: {act_del}")
                # print_debug(f"tot_del: {tot_del}")
                # print_debug(f"nstep_len before: {nstep_len}")
                nstep_len = torch.where(((tot_del <= i) & (tot_del < nstep_len) | done), ones_tens * i, nstep_len)
                # print_debug(f"nstep_len after: {nstep_len}")
            print_debug(f"nstep_len: {nstep_len}")
            nstep_max_len = torch.max(nstep_len)
            assert nstep_max_len < self.act_buf_size, "Delays longer than the action buffer (e.g. infinite) are not supported"
            print_debug(f"nstep_max_len: {nstep_max_len}")
            nstep_one_hot = torch.zeros(len(nstep_len), nstep_max_len + 1, device=self.device, requires_grad=False).scatter_(1, nstep_len.unsqueeze(1), 1.)
            # print_debug(f"nstep_one_hot: {nstep_one_hot}")
        else:  # RTAC is equivalent to doing only 1-step backups (i.e. nstep_len==0)
            nstep_len = torch.zeros(batch_size, device=self.device, dtype=int_tens_type, requires_grad=False)
            nstep_max_len = torch.max(nstep_len)
            nstep_one_hot = torch.zeros(len(nstep_len), nstep_max_len + 1, device=self.device, requires_grad=False).scatter_(1, nstep_len.unsqueeze(1), 1.)
            # terminals = terminals if self.act_buf_size == 1 else terminals * 0.0  # the way the replay memory works, RTAC will never encounter terminal states for buffers of more than 1 action
        # print_debug(f"nstep_len: {nstep_len}")
        # print_debug(f"nstep_max_len: {nstep_max_len}")
        # print_debug(f"nstep_one_hot: {nstep_one_hot}")

        # we compute the terminals tensor here
        terminals = torch.where((done_traj[nstep_len + 1]), ones_tens, zeros_tens)

        print_debug(f"terminals: {terminals}")
        print("DEBUG EXIT")
        exit()

        # use the current policy to compute a new trajectory of actions of length self.act_buf_size
        for i in range(self.act_buf_size + 1):
            # compute a new action and update the corresponding *next* augmented observation:
            augm_obs = augm_obs_traj[i]  # FIXME: this modifies augm_obs_traj, check that this is not an issue
            # print_debug(f"augm_obs at index {i}: {augm_obs}")
            if i > 0:
                # FIXME: check that this won't mess with autograd
                # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                act_slice = tuple(self.traj_new_actions[self.act_buf_size - i:self.act_buf_size])  # FIXME: check that first action in the action buffer is indeed the last computed action
                augm_obs = augm_obs[:1] + ((act_slice + augm_obs[1][i:]), ) + augm_obs[2:]
                # print_debug(f"augm_obs at index {i} after replacing actions: {augm_obs}")
            if i < self.act_buf_size:  # we don't compute the action for the last observation of the trajectory
                new_action_distribution = self.model.actor(augm_obs)
                # this is stored in right -> left order for replacing correctly in augm_obs:
                self.traj_new_actions[self.act_buf_size - i - 1] = new_action_distribution.rsample()
                self.traj_new_actions_detach[self.act_buf_size - i - 1] = self.traj_new_actions[self.act_buf_size - i - 1].detach()
                # print_debug(f"self.traj_new_actions[self.act_buf_size - i - 1]: {self.traj_new_actions[self.act_buf_size - i - 1]}")
                # this is stored in left -> right order for to be consistent with the reward trajectory:
                self.traj_new_actions_log_prob[i] = new_action_distribution.log_prob(self.traj_new_actions[self.act_buf_size - i - 1])
                self.traj_new_actions_log_prob_detach[i] = self.traj_new_actions_log_prob[i].detach()
                # print_debug(f"self.traj_new_actions_log_prob[i]: {self.traj_new_actions_log_prob[i]}")
            # this is stored in left -> right order:
            self.traj_new_augm_obs[i] = augm_obs
        # print_debug(f"self.traj_new_actions: {self.traj_new_actions}")
        # print_debug(f"self.traj_new_actions_log_prob: {self.traj_new_actions_log_prob}")
        # print_debug(f"self.traj_new_augm_obs: {self.traj_new_augm_obs}")

        # We now compute the state-value estimate of the augmented states at which the computed actions will be applied for each trajectory of the batch
        # (caution: this can be a different position in the trajectory for each element of the batch).

        # We expect each augmented state to be of shape (obs:tensor, act_buf:(tensor, ..., tensor), obs_del:tensor, act_del:tensor). Each tensor is batched.
        # We want to execute only 1 forward pass in the state-value estimator, therefore we recreate an artificially batched augmented state for this specific purpose.

        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # print_debug(f"nstep_len: {nstep_len}")
        obs_s = torch.stack([self.traj_new_augm_obs[i + 1][0][ibatch] for ibatch, i in enumerate(nstep_len)])
        act_s = tuple(torch.stack([self.traj_new_augm_obs[i + 1][1][iact][ibatch] for ibatch, i in enumerate(nstep_len)]) for iact in range(self.old_act_buf_size))
        od_s = torch.stack([self.traj_new_augm_obs[i + 1][2][ibatch] for ibatch, i in enumerate(nstep_len)])
        ad_s = torch.stack([self.traj_new_augm_obs[i + 1][3][ibatch] for ibatch, i in enumerate(nstep_len)])
        mod_augm_obs = tuple((obs_s, act_s, od_s, ad_s))
        # print_debug(f"mod_augm_obs: {mod_augm_obs}")

        # print_debug(" --- CRITIC LOSS ---")

        with torch.no_grad():

            # These are the delayed state-value estimates we are looking for:
            target_mod_val = [c(mod_augm_obs) for c in self.model_target.critics]
            # print_debug(f"target_mod_val of all critics: {target_mod_val}")
            target_mod_val = reduce(torch.min, torch.stack(target_mod_val)).squeeze()  # minimum target estimate
            # print_debug(f"target_mod_val before removing terminal states: {target_mod_val}")
            target_mod_val = target_mod_val * (1. - terminals)
            # print_debug(f"target_mod_val after removing terminal states: {target_mod_val}")

            # Now let us use this to compute the state-value targets of the batch of initial augmented states:

            value_target = torch.zeros(batch_size, device=self.device)
            backup_started = torch.zeros(batch_size, device=self.device)
            # print_debug(f"self.discount: {self.discount}")
            # print_debug(f"self.reward_scale: {self.reward_scale}")
            # print_debug(f"self.entropy_scale: {self.entropy_scale}")
            # print_debug(f"terminals: {terminals}")
            for i in reversed(range(nstep_max_len + 1)):
                start_backup_mask = nstep_one_hot[:, i]
                backup_started += start_backup_mask
                # print_debug(f"i: {i}")
                # print_debug(f"start_backup_mask: {start_backup_mask}")
                # print_debug(f"backup_started: {backup_started}")
                value_target = self.reward_scale * rew_traj[i] - self.entropy_scale * self.traj_new_actions_log_prob_detach[i] + backup_started * self.discount * (value_target + start_backup_mask * target_mod_val)
                # print_debug(f"rew_traj[i]: {rew_traj[i]}")
                # print_debug(f"self.traj_new_actions_log_prob_detach[i]: {self.traj_new_actions_log_prob_detach[i]}")
                # print_debug(f"new value_target: {value_target}")
            # print_debug(f"state-value target: {value_target}")

        # end of torch.no_grad()

        assert values[0].shape == value_target.shape, f"values[0].shape : {values[0].shape} != value_target.shape : {value_target.shape}"
        assert not value_target.requires_grad

        # Now the critic loss is:

        loss_critic = sum(mse_loss(v, value_target) for v in values)
        # print_debug(f"loss_critic: {loss_critic}")

        # actor loss:
        # TODO: there is probably a way of merging this with the previous for loop

        # print_debug(" --- ACTOR LOSS ---")

        model_mod_val = [c(mod_augm_obs) for c in self.model_nograd.critics]
        # print_debug(f"model_mod_val of all critics: {model_mod_val}")
        model_mod_val = reduce(torch.min, torch.stack(model_mod_val)).squeeze()  # minimum model estimate
        # print_debug(f"model_mod_val before removing terminal states: {model_mod_val}")
        model_mod_val = model_mod_val * (1. - terminals)
        # print_debug(f"model_mod_val after removing terminal states: {model_mod_val}")

        loss_actor = torch.zeros(batch_size, device=self.device)
        backup_started = torch.zeros(batch_size, device=self.device)
        # print_debug(f"self.discount: {self.discount}")
        # print_debug(f"self.reward_scale: {self.reward_scale}")
        # print_debug(f"self.entropy_scale: {self.entropy_scale}")
        # print_debug(f"terminals: {terminals}")
        for i in reversed(range(nstep_max_len + 1)):
            start_backup_mask = nstep_one_hot[:, i]
            backup_started += start_backup_mask
            # print_debug(f"i: {i}")
            # print_debug(f"start_backup_mask: {start_backup_mask}")
            # print_debug(f"backup_started: {backup_started}")
            loss_actor = - self.entropy_scale * self.traj_new_actions_log_prob[i] + backup_started * self.discount * (loss_actor + start_backup_mask * model_mod_val)
            # print_debug(f"self.traj_new_actions_log_prob[i]: {self.traj_new_actions_log_prob[i]}")
            # print_debug(f"new negative loss_actor: {loss_actor}")
        loss_actor = - loss_actor.mean(0)
        # print_debug(f"final loss_actor: {loss_actor}")

        # update model
        self.optimizer.zero_grad()
        loss_total = self.loss_alpha * loss_actor + (1 - self.loss_alpha) * loss_critic
        loss_total.backward()
        self.optimizer.step()

        # update target model and normalizers
        exponential_moving_average(self.model_target.parameters(), self.model.parameters(), self.target_update)
        # exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)

        # assert False

        return dict(
            loss_total=loss_total.detach(),
            loss_critic=loss_critic.detach(),
            loss_actor=loss_actor.detach(),
            # outputnorm_reward_mean=self.outputnorm.mean[0],
            # outputnorm_entropy_mean=self.outputnorm.mean[-1],
            # outputnorm_reward_std=self.outputnorm.std[0],
            # outputnorm_entropy_std=self.outputnorm.std[-1],
            memory_size=len(self.memory),
            # entropy_scale=self.entropy_scale
        )


DrtacTraining = partial(
    Training,
    Agent=partial(
        Agent,
        rtac=False,  # set this to True for reverting to RTAC
        batchsize=128,
        Model=partial(
            Mlp,
            act_delay=True,
            obs_delay=True)),
    Env=partial(
        RandomDelayEnv,
        id="Pendulum-v0",
        min_observation_delay=0,
        sup_observation_delay=1,
        min_action_delay=0,
        sup_action_delay=1),
        # possible alternative values for the delays: [(0, 1, 0, 1), (0, 2, 0, 1), (0, 1, 0, 2), (1, 2, 1, 2), (0, 3, 0, 3)]
    )

DrtacTest = partial(
    Training,
    Agent=partial(
        Agent,
        rtac=True,  # set this to True for reverting to RTAC
        batchsize=128,
        start_training=256,
        Model=partial(
            Mlp,
            act_delay=True,
            obs_delay=True)),
    Env=partial(
        RandomDelayEnv,
        id="Pendulum-v0",
        min_observation_delay=0,
        sup_observation_delay=2,
        min_action_delay=0,
        sup_action_delay=2))

DrtacShortTimesteps = partial(  # works at 2/5 of the original Mujoco timescale
    DrtacTraining,
    Env=partial(frame_skip=2),  # only works with Mujoco tasks (for now)
    steps=5000,
    Agent=partial(memory_size=2500000, training_steps=2/5, start_training=25000, discount=0.996, entropy_scale=2/5)
)

# To compare against SAC:
DelayedSacTraining = partial(
    Training,
    Agent=partial(
        agents.sac.Agent,
        batchsize=128,
        Model=partial(
            agents.sac_models_rd.Mlp,
            act_delay=True,
            obs_delay=True),
        OutputNorm=partial(beta=0., zero_debias=False),
    ),
    Env=partial(
        RandomDelayEnv,
        id="Pendulum-v0",
        min_observation_delay=0,
        sup_observation_delay=1,
        min_action_delay=0,
        sup_action_delay=1,
    ),
)

DelayedSacShortTimesteps = partial(  # works at 2/5 of the original Mujoco timescale
    DelayedSacTraining,
    Env=partial(frame_skip=2),  # only works with Mujoco tasks (for now)
    steps=5000,
    Agent=partial(memory_size=2500000, training_steps=2/5, start_training=25000, discount=0.996, entropy_scale=2/5)
)


UndelayedSacTraining = partial(
    Training,
    Agent=partial(
        agents.sac_undelayed.Agent,
        batchsize=128,
        Model=partial(
            agents.sac_models_rd.Mlp,
            act_delay=True,
            obs_delay=True),
        OutputNorm=partial(beta=0., zero_debias=False),
    ),
    Env=partial(
        RandomDelayEnv,
        id="Pendulum-v0",
        min_observation_delay=0,
        sup_observation_delay=1,
        min_action_delay=0,
        sup_action_delay=1,
    ),
)


if __name__ == "__main__":
    from pandas.plotting import autocorrelation_plot

    from agents import run
    run(DrtacTraining)
