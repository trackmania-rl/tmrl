from dataclasses import InitVar, dataclass

import gym
import torch
import numpy as np
from torch.nn.functional import leaky_relu

from agents.util import collate, partition
from torch.nn import Linear, Sequential, ReLU, ModuleList, Module, Conv2d, MaxPool2d
from torch.nn import functional as F
from agents.nn import TanhNormalLayer, SacLinear, big_conv
from agents.envs import UntouchedGymEnv
import torch.nn as nn
from agents.memory_dataloading import MemoryTMNF, MemoryTM2020


class ActorModule(Module):
    device = 'cpu'
    actor: callable

    # noinspection PyMethodOverriding
    def to(self, device):
        """keeps track which device this module has been moved to"""
        self.device = device
        return super().to(device=device)

    def reset(self):
        """Initialize the hidden state. This will be collated before being fed to the actual model and thus should be a structure of numpy arrays rather than torch tensors."""
        return np.array(())  # just so we don't get any errors when collating and partitioning

    def act(self, state, obs, r, done, info, train=False):
        """allows this module to be used with gym.Env
        converts inputs to torch tensors and converts outputs to numpy arrays"""
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            action_distribution = self.actor(obs)
            action = action_distribution.sample() if train else action_distribution.sample_deterministic()
        action, = partition(action)
        return action, state, []


class MlpActionValue(Sequential):
    def __init__(self, dim_obs, dim_action, hidden_units):
        super().__init__(
            SacLinear(dim_obs + dim_action, hidden_units), ReLU(),
            SacLinear(hidden_units, hidden_units), ReLU(),
            Linear(hidden_units, 2)
        )

    # noinspection PyMethodOverriding
    def forward(self, obs, action):
        x = torch.cat((*obs, action), -1)
        return super().forward(x)


class MlpPolicy(Sequential):
    def __init__(self, dim_obs, dim_action, hidden_units):
        super().__init__(
            SacLinear(dim_obs, hidden_units), ReLU(),
            SacLinear(hidden_units, hidden_units), ReLU(),
            TanhNormalLayer(hidden_units, dim_action)
        )

    # noinspection PyMethodOverriding
    def forward(self, obs):
        return super().forward(torch.cat(obs, -1))  # XXX


class Mlp(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256, num_critics: int = 2):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple), f"{observation_space}"
        dim_obs = sum(space.shape[0] for space in observation_space)
        dim_action = action_space.shape[0]
        self.critics = ModuleList(MlpActionValue(dim_obs, dim_action, hidden_units) for _ in range(num_critics))
        self.actor = MlpPolicy(dim_obs, dim_action, hidden_units)
        self.critic_output_layers = [c[-1] for c in self.critics]


# === convolutional models =======================================================================================================
class ConvActor(Module):
    def __init__(self, observation_space, action_space, hidden_units: int = 512, Conv: type = big_conv):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        (img_sp, vec_sp), *aux = observation_space

        self.conv = Conv(img_sp.shape[0])
        with torch.no_grad():
            conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)
            print("conv_size =", conv_size)
        self.lin1 = torch.nn.Linear(
            conv_size + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux),
            hidden_units)
        self.lin2 = torch.nn.Linear(
            hidden_units + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux),
            hidden_units
        )
        self.output_layer = TanhNormalLayer(hidden_units, action_space.shape[0])

    def forward(self, observation):
        (x, vec), *aux = observation
        x = x.type(torch.float32)
        x = x / 255 - 0.5
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = leaky_relu(self.lin1(torch.cat((x, vec, *aux), -1)))
        x = leaky_relu(self.lin2(torch.cat((x, vec, *aux), -1)))
        x = self.output_layer(x)
        return x


class ConvCritic(Module):
    def __init__(self, observation_space, action_space, hidden_units: int = 512, Conv: type = big_conv):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        (img_sp, vec_sp), *aux = observation_space

        self.conv = Conv(img_sp.shape[0])
        with torch.no_grad():
            conv_size = self.conv(torch.zeros((1, *img_sp.shape))).view(1, -1).size(1)

        self.lin1 = torch.nn.Linear(
            conv_size + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux) + action_space.shape[0],
            hidden_units)
        self.lin2 = torch.nn.Linear(
            hidden_units + vec_sp.shape[0] + sum(sp.shape[0] for sp in aux) + action_space.shape[0],
            hidden_units
        )
        self.output_layer = torch.nn.Linear(hidden_units, 2)
        self.critic_output_layers = self.output_layer,

    def forward(self, observation, a):
        (x, vec), *aux = observation
        x = x.type(torch.float32)
        x = x / 255 - 0.5
        x = self.conv(x)
        print(x.shape)
        # x = x.view(x.size(0), -1)
        x = leaky_relu(self.lin1(torch.cat((x, vec, *aux, a), -1)))
        x = leaky_relu(self.lin2(torch.cat((x, vec, *aux, a), -1)))
        x = self.output_layer(x)
        return x


class ConvModel(ActorModule):
    def __init__(self, observation_space, action_space, num_critics: int = 2, hidden_units: int = 256, Conv: type = big_conv):
        super().__init__()
        self.actor = ConvActor(observation_space, action_space, hidden_units, Conv)
        self.critics = ModuleList(ConvCritic(observation_space, action_space, hidden_units, Conv) for _ in range(num_critics))
        self.critic_output_layers = sum((c.critic_output_layers for c in self.critics), ())


# === Testing ==========================================================================================================
class TestMlp(ActorModule):
    def act(self, state, obs, r, done, info, train=False):
        return obs.copy(), state, {}


# === Trackmania =======================================================================================================

class TMModule1(Module):
    def __init__(self, observation_space, action_space, is_q_network):  # FIXME: action_space param is useless
        """
        Args:
        """
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)

        torch.autograd.set_detect_anomaly(True)

        # print(f"DEBUG: observation_space:{observation_space}")
        # print(f"DEBUG: action_space:{action_space}")
        # print(f"DEBUG: is_q_network:{is_q_network}")
        self.img_dims = observation_space[1].shape
        # print(f"DEBUG: self.img_dims:{self.img_dims}")
        self.vel_dim = observation_space[0].shape[0]
        # print(f"DEBUG: self.vel_dim:{self.vel_dim}")

        self.is_q_network = is_q_network

        self.act_dim = action_space.shape[0]
        # print(f"DEBUG: self.act_dim:{self.act_dim}")

        self.conv1 = Conv2d(3, 6, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)

        if self.is_q_network:
            self.fc1 = Linear(16 * 5 * 5 + self.vel_dim + self.act_dim, 120)  # 6*6 from image dimension
            # self.fc2 = Linear(120, 84)
            # self.fc3 = Linear(84, 2)
        else:
            self.fc1 = Linear(16 * 5 * 5 + self.vel_dim, 120)
            # self.fc2 = Linear(120, 84)
            # self.fc3 = TanhNormalLayer(84, self.act_dim)

    def forward(self, x):
        # assert isinstance(x, tuple), f"x is not a tuple: {x}"
        vel = x[0].float()
        im = x[1].float()[:, 0]
        im = self.pool(F.relu(self.conv1(im)))
        im = self.pool(F.relu(self.conv2(im)))
        im = im.view(-1, 16 * 5 * 5)
        if self.is_q_network:
            act = x[2].float()
            h = torch.cat((im, vel, act), dim=1)
        else:
            h = torch.cat((im, vel), dim=1)
        h = self.fc1(h)
        return h


class TMModuleNet(Module):
    def __init__(self, observation_space, action_space, is_q_network):  # FIXME: action_space param is useless
        """
        Args:
        """
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)

        torch.autograd.set_detect_anomaly(True)

        # print(f"DEBUG: observation_space:{observation_space}")
        # print(f"DEBUG: action_space:{action_space}")
        # print(f"DEBUG: is_q_network:{is_q_network}")
        self.img_dims = observation_space[1].shape
        # print(f"DEBUG: self.img_dims:{self.img_dims}")
        self.vel_dim = observation_space[0].shape[0]
        # print(f"DEBUG: self.vel_dim:{self.vel_dim}")

        self.is_q_network = is_q_network

        self.act_dim = action_space.shape[0]
        # print(f"DEBUG: self.act_dim:{self.act_dim}")

        self.conv1 = Conv2d(3, 6, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)

        if self.is_q_network:
            self.fc1 = Linear(2320 + self.vel_dim + self.act_dim, 120)  # 6*6 from image dimension
            # self.fc2 = Linear(120, 84)
            # self.fc3 = Linear(84, 2)
        else:
            self.fc1 = Linear(2320 + self.vel_dim, 120)
            # self.fc2 = Linear(120, 84)
            # self.fc3 = TanhNormalLayer(84, self.act_dim)

    def forward(self, x):
        # assert isinstance(x, tuple), f"x is not a tuple: {x}"
        vel = x[0].float()
        im1 = x[1].float()[:, 0]
        im2 = x[1].float()[:, 1]
        im3 = x[1].float()[:, 2]
        im4 = x[1].float()[:, 3]
        im = torch.cat((im1, im2, im3, im4), dim=2)  # TODO : check device

        # print(f"DEBUG: x[1].shape:{x[1].shape}")
        # print(f"DEBUG: im1.shape:{im1.shape}")
        # print(f"DEBUG: im.shape:{im.shape}")

        # size :
        # input: (3 * (32*4) * 32)
        # (3 * 128 * 32)
        # cv1 (3, 6, 5): (6 * 124 * 18)
        # pool (2): (6 * 62 * 9)
        # cv2 (6, 16, 5): (16 * 58 * 5)
        # pool (2): (16 * 29 * 3)

        im = self.pool(F.relu(self.conv1(im)))
        im = self.pool(F.relu(self.conv2(im)))
        im = im.view(-1, 2320)
        if self.is_q_network:
            act = x[2].float()
            h = torch.cat((im, vel, act), dim=1)
        else:
            h = torch.cat((im, vel), dim=1)
        h = self.fc1(h)
        return h


class TMModuleResnet(Module):
    def __init__(self, observation_space, action_space, is_q_network, act_in_obs=False):  # FIXME: action_space param is useless
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        torch.autograd.set_detect_anomaly(True)
        self.img_dims = observation_space[1].shape
        self.vel_dim = observation_space[0].shape[0]
        self.is_q_network = is_q_network
        self.act_in_obs = act_in_obs
        self.act_dim = action_space.shape[0]
        self.cnn = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 200)  # remove the last fc layer
        dim_fc1 = 200 + self.vel_dim
        if self.is_q_network:
            # print(f"DEBUG: created with q")
            dim_fc1 += self.act_dim
        if self.act_in_obs:
            # print(f"DEBUG: created with act_in_obs")
            dim_fc1 += self.act_dim
        self.fc1 = Linear(dim_fc1, 120)

    def forward(self, x):
        assert isinstance(x, tuple), f"x is not a tuple: {x}"
        # print(f"DEBUG: len(x):{len(x)}")
        # print(f"DEBUG: x[0]:{x[0]}")
        vel = x[0].float()
        im1 = x[1].float()[:, 0]
        im2 = x[1].float()[:, 1]
        im3 = x[1].float()[:, 2]
        im4 = x[1].float()[:, 3]
        if self.act_in_obs:
            prev_act = x[2].float()
            # print(f"DEBUG: forward act_in_obs")
            # print(f"DEBUG: prev_act:{prev_act}")
        im = torch.cat((im1, im2, im3, im4), dim=2)  # TODO : check device
        im = self.cnn(im)
        if self.is_q_network:
            act = x[-1].float()
            # print(f"DEBUG: forward q net")
            # print(f"DEBUG: act:{act}")
            # print(f"DEBUG1 im.shape{im.shape}, vel.shape{vel.shape}, act.shape{act.shape}")
            h = torch.cat((im, vel, prev_act, act), dim=1) if self.act_in_obs else torch.cat((im, vel, act), dim=1)
        else:
            # print(f"DEBUG2 im.shape : {im.shape}, vel.shape : {vel.shape} , prev_act.shape : {prev_act.shape}")
            h = torch.cat((im, vel, prev_act), dim=1) if self.act_in_obs else torch.cat((im, vel), dim=1)
        #print(f"DEBUG h.shape{h.shape}")
        h = self.fc1(h)
        return h


class TMActionValue(Sequential):
    def __init__(self, observation_space, action_space, act_in_obs=False):
        super().__init__(
            TMModuleResnet(observation_space, action_space, is_q_network=True, act_in_obs=act_in_obs), ReLU(),
            Linear(120, 84), ReLU(),
            Linear(84, 2)  # we separate reward components
        )

    # noinspection PyMethodOverriding
    def forward(self, obs, action):
        x = (*obs, action)
        res = super().forward(x)
        # print(f"DEBUG: av res:{res}")
        return res


class TMPolicy(Sequential):
    def __init__(self, observation_space, action_space, act_in_obs=False):
        super().__init__(
            TMModuleResnet(observation_space, action_space, is_q_network=False, act_in_obs=act_in_obs), ReLU(),
            Linear(120, 84), ReLU(),
            TanhNormalLayer(84, action_space.shape[0])
        )

    # noinspection PyMethodOverriding
    def forward(self, obs):
        # res = super().forward(torch.cat(obs, 1))
        res = super().forward(obs)
        # print(f"DEBUG: po res:{res}")
        return res


class Tm_hybrid_1(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256, num_critics: int = 2, act_in_obs=False):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple), f"{observation_space} is not a spaces.Tuple"
        self.critics = ModuleList(TMActionValue(observation_space, action_space, act_in_obs=act_in_obs) for _ in range(num_critics))
        self.actor = TMPolicy(observation_space, action_space, act_in_obs=act_in_obs)
        self.critic_output_layers = [c[-1] for c in self.critics]


if __name__ == "__main__":
    from agents import TrainingOffline, run_wandb_tm
    from agents.util import partial
    from agents.sac import Agent
    import gym
    from gym_tmrl.envs.tmrl_env import TMInterface, TM2020Interface
    from agents.envs import UntouchedGymEnv
    from requests import get

    # trackmania agent (Yann):

    CHECKPOINT_PATH = r"D:\cp\exp0" #r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\chk\exp0"
    DATASET_PATH = r"D:\data2020" #r"C:\Users\Yann\Desktop\git\tmrl\data"
    ACT_IN_OBS = True
    BENCHMARK = False

    CONFIG_DICT = {
        #"interface": TMInterface,
        "interface": TM2020Interface,
        "time_step_duration": 0.05,
        "start_obs_capture": 0.04,
        "time_step_timeout_factor": 1.0,
        "ep_max_length": np.inf,
        "real_time": True,
        "async_threading": True,
        "act_in_obs": ACT_IN_OBS,
        "benchmark": BENCHMARK
    }

    Sac_tm = partial(
        TrainingOffline,
        Env=partial(UntouchedGymEnv,
                    id="gym_tmrl:gym-tmrl-v0",
                    gym_kwargs={"config": CONFIG_DICT}),
        epochs=5,
        rounds=1,
        steps=1,
        update_model_interval=1,
        update_buffer_interval=1,
        Agent=partial(Agent,
                      Memory=partial(MemoryTM2020,  # MemoryTMNF
                                     path_loc=DATASET_PATH,
                                     imgs_obs=4,
                                     act_in_obs=ACT_IN_OBS,
                                     ),
                      device='cuda',
                      Model=partial(Tm_hybrid_1,
                                    act_in_obs=ACT_IN_OBS),
                      memory_size=1000000,
                      batchsize=8,
                      lr=0.0003,  # default 0.0003
                      discount=0.99,
                      target_update=0.005,
                      reward_scale=5.0,
                      entropy_scale=1.0),
    )

    print("--- NOW RUNNING: SAC trackmania ---")
    public_ip = get('http://api.ipify.org').text
    run_wandb_tm(None, None, None, run_cls=Sac_tm, checkpoint_path=CHECKPOINT_PATH)
    #run(Sac_tm, checkpoint_path=r"C:/Users/Yann/Desktop/git/tmrl/checkpoint/exp")  # checkpoint
