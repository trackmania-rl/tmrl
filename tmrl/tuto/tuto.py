import random
import numpy as np
import torch
from torch.optim import Adam
from copy import deepcopy

from threading import Thread

from tmrl.networking import Server, RolloutWorker, Trainer
from tmrl.util import partial, cached_property
from tmrl.envs import GenericGymEnv

from tmrl.actor import TorchActorModule
from tmrl.util import prod

import tmrl.config.config_constants as cfg
from tmrl.training_offline import TorchTrainingOffline
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad

from tuto_envs.dummy_rc_drone_interface import DUMMY_RC_DRONE_CONFIG


CRC_DEBUG = False

# === Networking parameters ============================================================================================

security = None
password = cfg.PASSWORD

server_ip = "127.0.0.1"
server_port = 6666


# === Server ===========================================================================================================

if __name__ == "__main__":
    my_server = Server(security=security, password=password, port=server_port)


# === Environment ======================================================================================================

# rtgym interface:

my_config = DUMMY_RC_DRONE_CONFIG

# Environment class:

env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_config})


# Observation and action space:

dummy_env = env_cls()
act_space = dummy_env.action_space
obs_space = dummy_env.observation_space

print(f"action space: {act_space}")
print(f"observation space: {obs_space}")


# === Worker ===========================================================================================================

import torch.nn.functional as F

# ActorModule:

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=torch.nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return torch.nn.Sequential(*layers)


class MyActorModule(TorchActorModule):
    """
    Directly adapted from the Spinup implementation of SAC
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU):
        super().__init__(observation_space, action_space)
        dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = torch.nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = torch.nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        net_out = self.net(torch.cat(obs, -1))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = torch.distributions.normal.Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        pi_action = pi_action.squeeze()
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return a.cpu().numpy()


actor_module_cls = partial(MyActorModule)


# Sample compression

def my_sample_compressor(prev_act, obs, rew, terminated, truncated, info):
    """
    Compresses samples before sending over network.

    This function creates the sample that will actually be stored in local buffers for networking.
    This is to compress the sample before sending it over the Internet/local network.
    Buffers of such samples will be given as input to the append() method of the memory.
    When you implement such compressor, you must implement a corresponding decompressor.
    This decompressor is the append() or get_transition() method of the memory.

    Args:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, terminated, truncated, info: outcome of the transition
    Returns:
        prev_act_mod: compressed prev_act
        obs_mod: compressed obs
        rew_mod: compressed rew
        terminated_mod: compressed terminated
        truncated_mod: compressed truncated
        info_mod: compressed info
    """
    prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod = prev_act, obs, rew, terminated, truncated, info
    obs_mod = obs_mod[:4]  # here we remove the action buffer from observations
    return prev_act_mod, obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod


sample_compressor = my_sample_compressor


# Device

device = "cpu"


# Networking

max_samples_per_episode = 1000


# Model files

my_run_name = "tutorial"
weights_folder = cfg.WEIGHTS_FOLDER

model_path = str(weights_folder / (my_run_name + ".tmod"))
model_path_history = str(weights_folder / (my_run_name + "_"))
model_history = 10


# Instantiation of the RolloutWorker object:

if __name__ == "__main__":
    my_worker = RolloutWorker(
        env_cls=env_cls,
        actor_module_cls=actor_module_cls,
        sample_compressor=sample_compressor,
        device=device,
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        max_samples_per_episode=max_samples_per_episode,
        model_path=model_path,
        model_path_history=model_path_history,
        model_history=model_history,
        crc_debug=CRC_DEBUG)

    # my_worker.run(test_episode_interval=10)  # this would block the script here!


# === Trainer ==========================================================================================================

# --- Networking and files ---

weights_folder = cfg.WEIGHTS_FOLDER  # path to the weights folder
checkpoints_folder = cfg.CHECKPOINTS_FOLDER
my_run_name = "tutorial"

model_path = str(weights_folder / (my_run_name + "_t.tmod"))
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.tcpt"))

# --- TrainingOffline ---

# Dummy environment:

env_cls = partial(GenericGymEnv, id="real-time-gym-ts-v1", gym_kwargs={"config": my_config})
# env_cls = (observation_space, action_space)


# Memory:

from tmrl.memory import TorchMemory


def last_true_in_list(li):
    """
    Returns the index of the last True element in list li, or None.
    """
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None


class MyMemory(TorchMemory):
    def __init__(self,
                 act_buf_len=None,
                 device=None,
                 nb_steps=None,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=32,
                 dataset_path=""):

        self.act_buf_len = act_buf_len  # length of the action buffer

        super().__init__(device=device,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         crc_debug=CRC_DEBUG)

    def append_buffer(self, buffer):
        """
        buffer.memory is a list of compressed (act_mod, new_obs_mod, rew_mod, terminated_mod, truncated_mod, info_mod) samples
        """

        # decompose compressed samples into their relevant components:

        list_action = [b[0] for b in buffer.memory]
        list_x_position = [b[1][0] for b in buffer.memory]
        list_y_position = [b[1][1] for b in buffer.memory]
        list_x_target = [b[1][2] for b in buffer.memory]
        list_y_target = [b[1][3] for b in buffer.memory]
        list_reward = [b[2] for b in buffer.memory]
        list_terminated = [b[3] for b in buffer.memory]
        list_truncated = [b[4] for b in buffer.memory]
        list_info = [b[5] for b in buffer.memory]
        list_done = [b[3] or b[4] for b in buffer.memory]

        # append to self.data in some arbitrary way:

        if self.__len__() > 0:
            self.data[0] += list_action
            self.data[1] += list_x_position
            self.data[2] += list_y_position
            self.data[3] += list_x_target
            self.data[4] += list_y_target
            self.data[5] += list_reward
            self.data[6] += list_terminated
            self.data[7] += list_info
            self.data[8] += list_truncated
            self.data[9] += list_done
        else:
            self.data.append(list_action)
            self.data.append(list_x_position)
            self.data.append(list_y_position)
            self.data.append(list_x_target)
            self.data.append(list_y_target)
            self.data.append(list_reward)
            self.data.append(list_terminated)
            self.data.append(list_info)
            self.data.append(list_truncated)
            self.data.append(list_done)

        # trim self.data in some arbitrary way when self.__len__() > self.memory_size:

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]

    def __len__(self):
        if len(self.data) == 0:
            return 0  # self.data is empty
        result = len(self.data[0]) - self.act_buf_len - 1
        if result < 0:
            return 0  # not enough samples to reconstruct the action buffer
        else:
            return result  # we can reconstruct that many samples

    def get_transition(self, item):
        """
        Args:
            item: int: indice of the transition that the Trainer wants to sample
        Returns:
            full transition: (last_obs, new_act, rew, new_obs, terminated, truncated, info)
        """
        while True:  # this enables modifying item in edge cases

            # if item corresponds to a transition from a terminal state to a reset state
            if self.data[9][item + self.act_buf_len - 1]:
                # this wouldn't make sense in RL, so we replace item by a neighbour transition
                if item == 0:  # if first item of the buffer
                    item += 1
                elif item == self.__len__() - 1:  # if last item of the buffer
                    item -= 1
                elif random.random() < 0.5:  # otherwise, sample randomly
                    item += 1
                else:
                    item -= 1

            idx_last = item + self.act_buf_len - 1  # index of previous observation
            idx_now = item + self.act_buf_len  # index of new observation

            # rebuild the action buffer of both observations:
            actions = self.data[0][item:(item + self.act_buf_len + 1)]
            last_act_buf = actions[:-1]  # action buffer of previous observation
            new_act_buf = actions[1:]  # action buffer of new observation

            # correct the action buffer when it goes over a reset transition:
            # (NB: we have eliminated the case where the transition *is* the reset transition)
            eoe = last_true_in_list(self.data[9][item:(item + self.act_buf_len)])  # the last one is not important
            if eoe is not None:
                # either one or both action buffers are passing over a reset transition
                if eoe < self.act_buf_len - 1:
                    # last_act_buf is concerned
                    if item == 0:
                        # we have a problem: the previous action has been discarded; we cannot recover the buffer
                        # in this edge case, we randomly sample another item
                        item = random.randint(1, self.__len__())
                        continue
                    last_act_buf_eoe = eoe
                    # replace everything before last_act_buf_eoe by the previous action
                    prev_act = self.data[0][item - 1]
                    for idx in range(last_act_buf_eoe + 1):
                        act_tmp = last_act_buf[idx]
                        last_act_buf[idx] = prev_act
                        prev_act = act_tmp
                if eoe > 0:
                    # new_act_buf is concerned
                    new_act_buf_eoe = eoe - 1
                    # replace everything before new_act_buf_eoe by the previous action
                    prev_act = self.data[0][item]
                    for idx in range(new_act_buf_eoe + 1):
                        act_tmp = new_act_buf[idx]
                        new_act_buf[idx] = prev_act
                        prev_act = act_tmp

            # rebuild the previous observation:
            last_obs = (self.data[1][idx_last],  # x position
                        self.data[2][idx_last],  # y position
                        self.data[3][idx_last],  # x target
                        self.data[4][idx_last],  # y target
                        *last_act_buf)  # action buffer

            # rebuild the new observation:
            new_obs = (self.data[1][idx_now],  # x position
                       self.data[2][idx_now],  # y position
                       self.data[3][idx_now],  # x target
                       self.data[4][idx_now],  # y target
                       *new_act_buf)  # action buffer

            # other components of the transition:
            new_act = self.data[0][idx_now]  # action
            rew = np.float32(self.data[5][idx_now])  # reward
            terminated = self.data[6][idx_now]  # terminated signal
            truncated = self.data[8][idx_now]  # truncated signal
            info = self.data[7][idx_now]  # info dictionary

            break

        return last_obs, new_act, rew, new_obs, terminated, truncated, info


memory_cls = partial(MyMemory,
                     act_buf_len=my_config["act_buf_len"])


# Training agent:


class MyCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU):
        super().__init__()
        obs_dim = sum(prod(s for s in space.shape) for space in observation_space)
        act_dim = action_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)


class MyActorCriticModule(torch.nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=torch.nn.ReLU):
        super().__init__()
        self.actor = MyActorModule(observation_space, action_space, hidden_sizes, activation)  # our ActorModule :)
        self.q1 = MyCriticModule(observation_space, action_space, hidden_sizes, activation)  # Q network 1
        self.q2 = MyCriticModule(observation_space, action_space, hidden_sizes, activation)  # Q network 2


import itertools


class MyTrainingAgent(TrainingAgent):

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=MyActorCriticModule,  # an actor-critic module, encapsulating our ActorModule
                 gamma=0.99,  # discount factor
                 polyak=0.995,  # exponential averaging factor for the target critic
                 alpha=0.2,  # fixed (SAC v1) or initial (SAC v2) value of the entropy coefficient
                 lr_actor=1e-3,  # learning rate for the actor
                 lr_critic=1e-3,  # learning rate for the critic
                 lr_entropy=1e-3,  # entropy autotuning coefficient (SAC v2)
                 learn_entropy_coef=True,  # if True, SAC v2 is used, else, SAC v1 is used
                 target_entropy=None):  # if None, the target entropy for SAC v2 is set automatically
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        model = model_cls(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_entropy = lr_entropy
        self.learn_entropy_coef=learn_entropy_coef
        self.target_entropy = target_entropy
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)
        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)
        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        o, a, r, o2, d, _ = batch  # ignore the truncated signal
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
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        for p in self.q_params:
            p.requires_grad = False
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha_t * logp_pi - q_pi).mean()
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        for p in self.q_params:
            p.requires_grad = True
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        ret_dict = dict(
            loss_actor=loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )
        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()
        return ret_dict


training_agent_cls = partial(MyTrainingAgent,
                             model_cls=MyActorCriticModule,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.2,
                             lr_actor=1e-3,
                             lr_critic=1e-3,
                             lr_entropy=1e-3,
                             learn_entropy_coef=True,
                             target_entropy=None)


# Training parameters:

epochs = 10  # maximum number of epochs, usually set this to np.inf
rounds = 10  # number of rounds per epoch
steps = 1000  # number of training steps per round
update_buffer_interval = 100
update_model_interval = 100
max_training_steps_per_env_step = 2.0
start_training = 400
device = None


# Trainer instance:

training_cls = partial(
    TorchTrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device)

if __name__ == "__main__":
    my_trainer = Trainer(
        training_cls=training_cls,
        server_ip=server_ip,
        server_port=server_port,
        password=password,
        model_path=model_path,
        checkpoint_path=checkpoints_path)  # None for not saving training checkpoints


# Separate threads for running the RolloutWorker and Trainer:


def run_worker(worker):
    worker.run(test_episode_interval=10)


def run_trainer(trainer):
    trainer.run()


if __name__ == "__main__":
    daemon_thread_worker = Thread(target=run_worker, args=(my_worker, ), kwargs={}, daemon=True)
    daemon_thread_worker.start()  # start the worker daemon thread

    run_trainer(my_trainer)

    # the worker daemon thread will be killed here.
