# Delayed RTAC

# standard library imports
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce

# third-party imports
import pandas
import torch
from torch.nn.functional import mse_loss

# local imports
import tmrl.sac
from tmrl.drtac_models import Mlp
from tmrl.nn import exponential_moving_average, no_grad
from tmrl.util import partial
import logging


def print_debug(st):
    # return
    logging.debug(" " + st)


class DcacInterface:
    def get_total_delay_tensor_from_augm_obs_tuple_of_tensors(self, augm_obs_tuple_of_tensors):
        """
        Returns:
            tot_del_tensor: (batch, 1)
        """

        # return tot_del_tensor

        raise NotImplementedError

    def replace_act_buf_in_augm_obs_tuple_of_tensors(self, augm_obs_tuple_of_tensors, act_buf_tuple_of_tensors):
        """
        must return a tensor with replaced action buffer
        the actions in act_buf are from the most recent at idx 0 to the oldest at idx -1
        Args:
            augm_obs_tuple_of_tensors
            act_buf_tuple_of_tensors
        Returns:
            mod_augm_obs_tuple_of_tensors
        """

        # return mod_augm_obs_tensor

        raise NotImplementedError

    def get_act_buf_tuple_of_tensors_from_augm_obs_tuple_of_tensors(self, augm_obs_tuple_of_tensors):
        """
        the actions in act_buf are from the oldest at idx -1 to the most recent at idx 0
        Args:
            augm_obs_tuple_of_tensors
        Returns:
            act_buf_tuple_of_tensors
        """

        # return act_buf_tuple_of_tensors

        raise NotImplementedError

    def get_act_buf_size(self):
        """
        Returns:
            act_buf_size: int
        """

        # return act_buf_size

        raise NotImplementedError

    def get_constant_and_max_possible_delay(self):
        """
        Returns:
            is_constant: (bool) whether the delays are constant
            max_possible_delay: (int) value of the maximum possible delay (must be <= act_buf_size)
        """

        # return is_constant, max_possible_delay

        raise NotImplementedError


@dataclass(eq=0)
class Agent(tmrl.sac.SacAgent):
    Interface: type = DcacInterface
    Model: type = Mlp
    loss_alpha: float = 0.2

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space

        assert self.device is not None
        device = self.device  # or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.Model(observation_space, action_space)
        self.model = model.to(device)
        # print_debug(f"self.model:{self.model}")
        self.model_target = no_grad(deepcopy(self.model))

        self.interface = self.Interface()
        self.act_buf_size = self.interface.get_act_buf_size()
        is_constant, max_possible_delay = self.interface.get_constant_and_max_possible_delay()
        if is_constant:
            self.constant = max_possible_delay
        else:
            self.constant = 0
        self.max_possible_delay = max_possible_delay

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = self.Memory(device)
        self.traj_new_actions = [
            None,
        ] * self.max_possible_delay
        self.traj_new_actions_detach = [
            None,
        ] * self.max_possible_delay
        self.traj_new_actions_log_prob = [
            None,
        ] * self.max_possible_delay
        self.traj_new_actions_log_prob_detach = [
            None,
        ] * self.max_possible_delay
        self.traj_new_augm_obs = [
            None,
        ] * (self.max_possible_delay + 1)  # + 1 because the trajectory is obs0 -> rew1(obs0,act0) -> obs1 -> ...

        self.is_training = False

    def train(self):
        # TODO: remove requires_grad everywhere it should not be

        # sample a trajectory of length self.max_possible_delay
        augm_obs_traj, rew_traj, done_traj = self.memory.sample()
        # TODO: act_traj is useless, it could be removed from the replay memory

        batch_size = done_traj[0].shape[0]
        # print_debug(f"batch_size: {batch_size}")

        # print_debug(f"augm_obs_traj: {augm_obs_traj}")
        # print_debug(f"rew_traj: {rew_traj}")
        # print_debug(f"done_traj: {done_traj}")

        # print_debug(f"augm_obs_traj[0]: {augm_obs_traj[0]}")
        # value of the first augmented state:
        values = [c(augm_obs_traj[0]).squeeze() for c in self.model.critics]
        # print_debug(f"values: {values}")

        # to determine the length of the n-step backup, nstep_len is the time at which the currently computed action (== i) or any action that followed (< i) has been applied first:
        # when nstep_len is k (in 0..self.max_possible_delaye-1), it means that the action computed with the first augmented observation of the trajectory will have an effect k+1 steps later
        # (either it will be applied, or an action that follows it will be applied)
        int_tens_type = torch.int64
        ones_tens = torch.ones(batch_size, device=self.device, dtype=int_tens_type, requires_grad=False)
        zeros_tens = torch.zeros(batch_size, device=self.device, dtype=int_tens_type, requires_grad=False)

        nstep_len = ones_tens * self.max_possible_delay
        for i in reversed(range(self.max_possible_delay)):  # caution: we don't care about the delay of the first observation in the trajectory, but we care about the last one
            if self.constant:
                tot_del = ones_tens * (self.constant - 1)  # -1 because of the convention we use for action delays
            else:
                # print_debug(f"self.constant:{self.constant}")
                # obs_del = augm_obs_traj[i + 1][2]
                # act_del = augm_obs_traj[i + 1][3]
                # tot_del = obs_del + act_del
                tot_del = self.interface.get_total_delay_tensor_from_augm_obs_tuple_of_tensors(augm_obs_traj[i + 1]) - 1  # -1 because act delay...
            # print_debug(f"i + 1:{i + 1}")
            # print_debug(f"done_traj:{done_traj}")
            done = done_traj[i + 1]
            # print_debug(f"i + 1: {i + 1}")
            # print_debug(f"obs_del: {obs_del}")
            # print_debug(f"act_del: {act_del}")
            # print_debug(f"tot_del: {tot_del}")
            # print_debug(f"nstep_len before: {nstep_len}")
            nstep_len = torch.where((((tot_del <= i) & (tot_del < nstep_len)) | (done > 0.0)), ones_tens * i, nstep_len)  # FIXME: done ?
            # print_debug(f"nstep_len after: {nstep_len}")
        # print_debug(f"nstep_len: {nstep_len}")
        nstep_max_len = torch.max(nstep_len)
        assert nstep_max_len < self.max_possible_delay, "Delays longer than the maximum possible delay are not supported (please clip them)"
        # print_debug(f"nstep_max_len: {nstep_max_len}")
        nstep_one_hot = torch.zeros(len(nstep_len), nstep_max_len + 1, device=self.device, requires_grad=False).scatter_(1, nstep_len.unsqueeze(1), 1.)
        # print_debug(f"nstep_one_hot: {nstep_one_hot}")
        #
        # print_debug(f"nstep_max_len: {nstep_max_len}")
        # print_debug(f"nstep_one_hot: {nstep_one_hot}")

        # we compute the terminals tensor here

        # print_debug(f"nstep_len: {nstep_len}")
        # print_debug(f"done_traj: {done_traj}")
        terminals = torch.tensor([done_traj[i][ibatch] for ibatch, i in enumerate((nstep_len + 1).tolist())], device=self.device)
        # print_debug(f"terminals: {terminals}")

        # CAUTION: act_buf_len is not the max possible delay !
        # use the current policy to compute a new trajectory of actions of length self.act_buf_size
        for i in range(self.max_possible_delay + 1):
            # compute a new action and update the corresponding *next* augmented observation:
            augm_obs = augm_obs_traj[i]  # FIXME: this modifies augm_obs_traj, check that this is not an issue
            # print_debug(f"augm_obs at index {i}: {augm_obs}")
            if i > 0:  # we don't need to modify the first obs of the trajectory
                # FIXME: check that this won't mess with autograd
                act_slice = tuple(self.traj_new_actions[self.max_possible_delay - i:self.max_possible_delay])  # FIXME: check order
                # augm_obs = augm_obs[:1] + ((act_slice + augm_obs[1][i:]), ) + augm_obs[2:]
                act_buf = self.interface.get_act_buf_tuple_of_tensors_from_augm_obs_tuple_of_tensors(augm_obs)  # most recent action at idx 0, oldest at idx -1
                act_buf = (
                    *act_slice,
                    *act_buf[i:],
                )  # resampled actions with the most recently resampled at idx 0 in the buffer
                augm_obs = self.interface.replace_act_buf_in_augm_obs_tuple_of_tensors(augm_obs, act_buf)
                # print_debug(f"augm_obs at index {i} after replacing actions: {augm_obs}")
            if i < self.max_possible_delay:  # we don't need to compute the action for the last observation of the trajectory
                # in the action buffer, the most recent action is at idx 0, the oldest at idx -1
                new_action_distribution = self.model.actor(augm_obs)
                # this is stored in right -> left order for replacing correctly in augm_obs:  # FIXME: check order
                self.traj_new_actions[self.max_possible_delay - i - 1] = new_action_distribution.rsample()
                self.traj_new_actions_detach[self.max_possible_delay - i - 1] = self.traj_new_actions[self.max_possible_delay - i - 1].detach()
                # print_debug(f"self.traj_new_actions[self.max_possible_delay - i - 1]: {self.traj_new_actions[self.max_possible_delay - i - 1]}")
                # this is stored in left -> right order for to be consistent with the reward trajectory:
                self.traj_new_actions_log_prob[i] = new_action_distribution.log_prob(self.traj_new_actions[self.max_possible_delay - i - 1])
                self.traj_new_actions_log_prob_detach[i] = self.traj_new_actions_log_prob[i].detach()
                # print_debug(f"self.traj_new_actions_log_prob[i]: {self.traj_new_actions_log_prob[i]}")
            # this is stored in left -> right order:
            self.traj_new_augm_obs[i] = augm_obs
        # print_debug(f"self.traj_new_actions: {self.traj_new_actions}")
        # print_debug(f"self.traj_new_actions_log_prob: {self.traj_new_actions_log_prob}")
        # print_debug(f"self.traj_new_augm_obs: {self.traj_new_augm_obs}")

        # We now compute the state-value estimate of the augmented states at which the computed actions will be applied for each trajectory of the batch
        # (caution: this can be a different position in the trajectory for each element of the batch).

        # We want to execute only 1 forward pass in the state-value estimator, therefore we recreate an artificially batched augmented state for this specific purpose.

        # TODO
        # # print_debug(f"nstep_len: {nstep_len}")
        # obs_s = torch.stack([self.traj_new_augm_obs[i + 1][0][ibatch] for ibatch, i in enumerate(nstep_len)])
        # act_s = tuple(torch.stack([self.traj_new_augm_obs[i + 1][1][iact][ibatch] for ibatch, i in enumerate(nstep_len)]) for iact in range(self.act_buf_size))
        # od_s = torch.stack([self.traj_new_augm_obs[i + 1][2][ibatch] for ibatch, i in enumerate(nstep_len)])
        # ad_s = torch.stack([self.traj_new_augm_obs[i + 1][3][ibatch] for ibatch, i in enumerate(nstep_len)])
        # mod_augm_obs = tuple((obs_s, act_s, od_s, ad_s))

        # mod_augm_obs = tuple((torch.stack([self.traj_new_augm_obs[i + 1][itup][ibatch] for ibatch, i in enumerate(nstep_len)]) for itup in range(len(self.traj_new_augm_obs[0]))))

        # print_debug(f"mod_augm_obs: {mod_augm_obs}")

        # print_debug(" --- CRITIC LOSS ---")

        with torch.no_grad():

            # These are the delayed state-value estimates we are looking for:

            # target_mod_val = [c(mod_augm_obs) for c in self.model_target.critics]
            # print_debug(f"target_mod_val of all critics: {target_mod_val}")
            # target_mod_val = reduce(torch.min, torch.stack(target_mod_val)).squeeze()  # minimum target estimate
            # print_debug(f"target_mod_val before removing terminal states: {target_mod_val}")
            # print_debug(f"terminals.device:{terminals.device}")
            # print_debug(f"target_mod_val.device:{target_mod_val.device}")
            # target_mod_val = target_mod_val * (1. - terminals)
            # print_debug(f"target_mod_val after removing terminal states: {target_mod_val}")

            # print_debug(f"nstep_max_len: {nstep_max_len}")

            target_mod_vals = [reduce(torch.min, torch.stack([c(self.traj_new_augm_obs[i + 1]) for c in self.model_target.critics])).squeeze() * (1. - terminals) for i in range(nstep_max_len + 1)]

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
                # print_debug(f"target_mod_vals[i]: {target_mod_vals[i]}")
                # TODO:
                # value_target = self.reward_scale * rew_traj[i] - self.entropy_scale * self.traj_new_actions_log_prob_detach[i] + backup_started * self.discount * (value_target + start_backup_mask * target_mod_val)

                value_target = self.reward_scale * rew_traj[i] - self.entropy_scale * self.traj_new_actions_log_prob_detach[i] + backup_started * self.discount * (
                    value_target + start_backup_mask * target_mod_vals[i])

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

        # model_mod_val = [c(mod_augm_obs) for c in self.model_nograd.critics]
        # print_debug(f"model_mod_val of all critics: {model_mod_val}")
        # model_mod_val = reduce(torch.min, torch.stack(model_mod_val)).squeeze()  # minimum model estimate
        # print_debug(f"model_mod_val before removing terminal states: {model_mod_val}")
        # model_mod_val = model_mod_val * (1. - terminals)
        # print_debug(f"model_mod_val after removing terminal states: {model_mod_val}")

        model_mod_vals = [reduce(torch.min, torch.stack([c(self.traj_new_augm_obs[i + 1]) for c in self.model_nograd.critics])).squeeze() * (1. - terminals) for i in range(nstep_max_len + 1)]

        # target_mod_vals = [reduce(torch.min, torch.stack([c(self.traj_new_augm_obs[i + 1]) for c in self.model_target.critics])).squeeze() * (1. - terminals) for i in range(nstep_max_len + 1)]

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
            # loss_actor = - self.entropy_scale * self.traj_new_actions_log_prob[i] + backup_started * self.discount * (loss_actor + start_backup_mask * model_mod_val)
            loss_actor = -self.entropy_scale * self.traj_new_actions_log_prob[i] + backup_started * self.discount * (loss_actor + start_backup_mask * model_mod_vals[i])
            # print_debug(f"self.traj_new_actions_log_prob[i]: {self.traj_new_actions_log_prob[i]}")
            # print_debug(f"new negative loss_actor: {loss_actor}")
        loss_actor = -loss_actor.mean(0)
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
