from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT, DummyRCDrone
import gym.spaces as spaces
import numpy as np
import cv2
import torch
from torch.optim import Adam
from copy import deepcopy

from threading import Thread

from tmrl.networking import Server, RolloutWorker, Trainer
from tmrl.util import partial, cached_property
from tmrl.envs import GenericGymEnv

from tmrl.actor import ActorModule
from tmrl.util import prod

import tmrl.config.config_constants as cfg
from tmrl.training_offline import TrainingOffline
from tmrl.training import TrainingAgent
from tmrl.nn import copy_shared, no_grad


CRC_DEBUG = False

# === Server ===========================================================================================================

my_server = Server(min_samples_per_server_packet=100)


# === Environment ======================================================================================================

# rtgym interface:

class DummyRCDroneInterface(RealTimeGymInterface):

    def __init__(self):
        self.rc_drone = None
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.initialized = False
        self.blank_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        self.rendering_thread = Thread(target=self._rendering_thread, args=(), kwargs={}, daemon=True)

    def _rendering_thread(self):
        from time import sleep
        while True:
            sleep(0.1)
            self.render()

    def get_observation_space(self):
        pos_x_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        pos_y_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        tar_x_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        tar_y_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        return spaces.Tuple((pos_x_space, pos_y_space, tar_x_space, tar_y_space))

    def get_action_space(self):
        return spaces.Box(low=-2.0, high=2.0, shape=(2,))

    def get_default_action(self):
        return np.array([0.0, 0.0], dtype='float32')

    def send_control(self, control):
        vel_x = control[0]
        vel_y = control[1]
        self.rc_drone.send_control(vel_x, vel_y)

    def reset(self):
        if not self.initialized:
            self.rc_drone = DummyRCDrone()
            self.initialized = True
            self.rendering_thread.start()
        pos_x, pos_y = self.rc_drone.get_observation()
        self.target[0] = np.random.uniform(-0.5, 0.5)
        self.target[1] = np.random.uniform(-0.5, 0.5)
        return [pos_x, pos_y, self.target[0], self.target[1]]

    def get_obs_rew_done_info(self):
        pos_x, pos_y = self.rc_drone.get_observation()
        tar_x = self.target[0]
        tar_y = self.target[1]
        obs = [pos_x, pos_y, tar_x, tar_y]
        rew = -np.linalg.norm(np.array([pos_x, pos_y], dtype=np.float32) - self.target)
        done = rew > -0.01
        info = {}
        return obs, rew, done, info

    def wait(self):
        self.send_control(self.get_default_action())

    def render(self):
        image = self.blank_image.copy()
        pos_x, pos_y = self.rc_drone.get_observation()
        image = cv2.circle(img=image,
                           center=(int(pos_x * 200) + 250, int(pos_y * 200) + 250),
                           radius=10,
                           color=(255, 0, 0),
                           thickness=1)
        image = cv2.circle(img=image,
                           center=(int(self.target[0] * 200) + 250, int(self.target[1] * 200) + 250),
                           radius=5,
                           color=(0, 0, 255),
                           thickness=-1)
        cv2.imshow("Dummy RC drone", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


# rtgym configuration dictionary:

my_config = DEFAULT_CONFIG_DICT.copy()
my_config["interface"] = DummyRCDroneInterface
my_config["time_step_duration"] = 0.05
my_config["start_obs_capture"] = 0.05
my_config["time_step_timeout_factor"] = 1.0
my_config["ep_max_length"] = 100
my_config["act_buf_len"] = 4
my_config["reset_act_buf"] = False
my_config["benchmark"] = True
my_config["benchmark_polyak"] = 0.2


# Environment class:

env_cls = partial(GenericGymEnv, id="real-time-gym-v0", gym_kwargs={"config": my_config})


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


class MyActorModule(ActorModule):
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
            return a.numpy()


actor_module_cls = partial(MyActorModule)


# Sample compression

def my_sample_compressor(prev_act, obs, rew, done, info):
    """
    Compresses samples before sending over network.

    This function creates the sample that will actually be stored in local buffers for networking.
    This is to compress the sample before sending it over the Internet/local network.
    Buffers of such samples will be given as input to the append() method of the dataloading memory.
    When you implement such compressor, you must implement a corresponding decompressor.
    This decompressor is the append() or get_transition() method of the memory.

    Args:
        prev_act: action computed from a previous observation and applied to yield obs in the transition
        obs, rew, done, info: outcome of the transition
    Returns:
        prev_act_mod: compressed prev_act
        obs_mod: compressed obs
        rew_mod: compressed rew
        done_mod: compressed done
        info_mod: compressed info
    """
    prev_act_mod, obs_mod, rew_mod, done_mod, info_mod = prev_act, obs, rew, done, info
    obs_mod = obs_mod[:4]  # here we remove the action buffer from observations
    return prev_act_mod, obs_mod, rew_mod, done_mod, info_mod


sample_compressor = my_sample_compressor


# Device

device = "cpu"


# Networking

server_ip = "127.0.0.1"
min_samples_per_worker_packet = 100
max_samples_per_episode = 1000


# Model files

my_run_name = "tutorial"
weights_folder = cfg.WEIGHTS_FOLDER

model_path = str(weights_folder / (my_run_name + ".pth"))
model_path_history = str(weights_folder / (my_run_name + "_"))
model_history = 10


# Instantiation of the RolloutWorker object:

my_worker = RolloutWorker(
    env_cls=env_cls,
    actor_module_cls=actor_module_cls,
    sample_compressor=sample_compressor,
    device=device,
    server_ip=server_ip,
    min_samples_per_worker_packet=min_samples_per_worker_packet,
    max_samples_per_episode=max_samples_per_episode,
    model_path=model_path,
    model_path_history=model_path_history,
    model_history=model_history,
    crc_debug=CRC_DEBUG)

# my_worker.run(test_episode_interval=10)  # this would block the script here!


# === Trainer ==========================================================================================================

# --- Networking and files ---

server_ip = "127.0.0.1"

weights_folder = cfg.WEIGHTS_FOLDER  # path to the weights folder
checkpoints_folder = cfg.CHECKPOINTS_FOLDER
my_run_name = "tutorial"

model_path = str(weights_folder / (my_run_name + "_t.pth"))
checkpoints_path = str(checkpoints_folder / (my_run_name + "_t.cpt"))

# --- TrainingOffline ---

# Dummy environment:

env_cls = partial(GenericGymEnv, id="real-time-gym-v0", gym_kwargs={"config": my_config})
# env_cls = (observation_space, action_space)


# MemoryDataloading:

from tmrl.memory_dataloading import MemoryDataloading

class MyMemoryDataloading(MemoryDataloading):
    def __init__(self,
                 act_buf_len=None,
                 device=None,
                 nb_steps=None,
                 obs_preprocessor: callable = None,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=32,
                 dataset_path=""):

        self.act_buf_len = act_buf_len  # length of the action buffer

        super().__init__(device=device,
                         nb_steps=nb_steps,
                         obs_preprocessor=obs_preprocessor,
                         sample_preprocessor=sample_preprocessor,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         crc_debug=CRC_DEBUG)

    def append_buffer(self, buffer):
        """
        buffer.memory is a list of compressed (act_mod, new_obs_mod, rew_mod, done_mod, info_mod) samples
        """

        # decompose compressed samples into their relevant components:

        list_action = [b[0] for b in buffer.memory]
        list_x_position = [b[1][0] for b in buffer.memory]
        list_y_position = [b[1][1] for b in buffer.memory]
        list_x_target = [b[1][2] for b in buffer.memory]
        list_y_target = [b[1][3] for b in buffer.memory]
        list_reward = [b[2] for b in buffer.memory]
        list_done = [b[3] for b in buffer.memory]
        list_info = [b[4] for b in buffer.memory]

        # append to self.data in some arbitrary way:

        if self.__len__() > 0:
            self.data[0] += list_action
            self.data[1] += list_x_position
            self.data[2] += list_y_position
            self.data[3] += list_x_target
            self.data[4] += list_y_target
            self.data[5] += list_reward
            self.data[6] += list_done
            self.data[7] += list_info
        else:
            self.data.append(list_action)
            self.data.append(list_x_position)
            self.data.append(list_y_position)
            self.data.append(list_x_target)
            self.data.append(list_y_target)
            self.data.append(list_reward)
            self.data.append(list_done)
            self.data.append(list_info)

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
            full transition: (last_obs, new_act, rew, new_obs, done, info)
        """
        idx_last = item + self.act_buf_len - 1  # index of previous observation
        idx_now = item + self.act_buf_len  # index of new observation

        # rebuild the action buffer of both observations:
        actions = self.data[0][item:(item + self.act_buf_len + 1)]
        last_act_buf = actions[:-1]  # action buffer of previous observation
        new_act_buf = actions[1:]  # action buffer of new observation

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
        done = self.data[6][idx_now]  # done signal
        info = self.data[7][idx_now]  # info dictionary

        return last_obs, new_act, rew, new_obs, done, info


memory_cls = partial(MyMemoryDataloading,
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
        o, a, r, o2, d = batch
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
            loss_actor=loss_pi.detach(),
            loss_critic=loss_q.detach(),
        )
        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach()
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
    TrainingOffline,
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

my_trainer = Trainer(
    training_cls=training_cls,
    server_ip=server_ip,
    model_path=model_path,
    checkpoint_path=checkpoints_path)  # None for not saving training checkpoints


# Separate threads for running the RolloutWorker and Trainer:


def run_worker(worker):
    worker.run(test_episode_interval=10)


def run_trainer(trainer):
    trainer.run()


daemon_thread_worker = Thread(target=run_worker, args=(my_worker, ), kwargs={}, daemon=True)
daemon_thread_worker.start()  # start the worker daemon thread

run_trainer(my_trainer)

# the worker daemon thread will be killed here.
