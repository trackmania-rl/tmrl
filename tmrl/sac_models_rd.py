import gym
import torch

from torch.nn import Linear, Sequential, ReLU, ModuleList, Module
from torch.nn import functional as F
from tmrl.sac_models import ActorModule
from tmrl.nn import TanhNormalLayer

from tmrl.envs import RandomDelayEnv


class DelayedMlpModule(Module):
    def __init__(self, observation_space, action_space, is_Q_network, hidden_units: int = 256, obs_delay=True, act_delay=True, tbmdp=False):  # FIXME: action_space param is useless
        """
        Args:
            observation_space:
                Tuple((
                    obs_space,  # most recent observation
                    Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
                    Discrete(obs_delay_range.stop),  # observation delay int64
                    Discrete(act_delay_range.stop),  # action delay int64
                ))
            action_space
            is_Q_network: bool: if True, the input of forward() expects the action to be appended at the end of the input
            hidden_units: number of output units of this module
            (optional) obs_delay: bool (default True): if False, the observation delay of observation_space will be ignored (e.g. unknown)
            (optional) act_delay: bool (default True): if False, the action delay of observation_space will be ignored (e.g. unknown)
        """
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        # TODO: check that it is actually an instance of:
        # Tuple((
        # 	obs_space,  # most recent observation
        # 	Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop - 1)),  # action buffer
        # 	Discrete(obs_delay_range.stop),  # observation delay int64
        # 	Discrete(act_delay_range.stop),  # action delay int64
        # ))

        self.tbmdp = tbmdp
        self.is_Q_network = is_Q_network
        self.act_delay = act_delay
        self.obs_delay = obs_delay

        self.obs_dim = observation_space[0].shape[0]
        self.buf_size = len(observation_space[1])
        print(f"DEBUG: MLP self.buf_size: {self.buf_size}")
        self.act_dim = observation_space[1][0].shape[0]
        assert self.act_dim == action_space.shape[0], f"action spaces mismatch: {self.act_dim} and {action_space.shape[0]}"

        if self.is_Q_network:
            if self.tbmdp:
                self.lin = Linear(self.obs_dim + self.act_dim, hidden_units)
            elif self.act_delay and self.obs_delay:
                self.lin = Linear(self.obs_dim + (self.act_dim + 2) * self.buf_size + self.act_dim, hidden_units)
            elif self.act_delay or self.obs_delay:
                self.lin = Linear(self.obs_dim + (self.act_dim + 1) * self.buf_size + self.act_dim, hidden_units)
            else:
                self.lin = Linear(self.obs_dim + self.act_dim * self.buf_size + self.act_dim, hidden_units)
        else:
            if self.tbmdp:
                self.lin = Linear(self.obs_dim, hidden_units)
            elif self.act_delay and self.obs_delay:
                self.lin = Linear(self.obs_dim + (self.act_dim + 2) * self.buf_size, hidden_units)
            elif self.act_delay or self.obs_delay:
                self.lin = Linear(self.obs_dim + (self.act_dim + 1) * self.buf_size, hidden_units)
            else:
                self.lin = Linear(self.obs_dim + self.act_dim * self.buf_size, hidden_units)

    def forward(self, x):
        assert isinstance(x, tuple), f"x is not a tuple: {x}"
        # TODO: check that x is actually in:
        # Tuple((
        # 	obs_space,  # most recent observation
        # 	Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
        # 	Discrete(obs_delay_range.stop),  # observation delay int64
        # 	Discrete(act_delay_range.stop),  # action delay int64
        # ))

        # TODO: double check that everything is correct (dims, devices, autograd)
        # TODO: triple check devices...

        obs = x[0]

        if self.tbmdp:
            input = obs
            if self.is_Q_network:
                act = x[4]
                input = torch.cat((input, act), dim=1)
            h = self.lin(input)
            return h

        act_buf = torch.cat(x[1], dim=1)

        input = torch.cat((obs, act_buf), dim=1)

        batch_size = obs.shape[0]
        if self.obs_delay:
            obs_del = x[2]
            obs_one_hot = torch.zeros(batch_size, self.buf_size, device=input.device).scatter_(1, obs_del.unsqueeze(1), 1.0)
            input = torch.cat((input, obs_one_hot), dim=1)
        if self.act_delay:
            act_del = x[3]
            act_one_hot = torch.zeros(batch_size, self.buf_size, device=input.device).scatter_(1, act_del.unsqueeze(1), 1.0)
            input = torch.cat((input, act_one_hot), dim=1)
        if self.is_Q_network:
            act = x[4]
            input = torch.cat((input, act), dim=1)

        h = self.lin(input)

        return h


class MlpActionValue(Sequential):
    def __init__(self, observation_space, action_space, hidden_units, act_delay=True, obs_delay=True, tbmdp=False):
        super().__init__(
            DelayedMlpModule(observation_space, action_space, is_Q_network=True, act_delay=act_delay, obs_delay=obs_delay, tbmdp=tbmdp), ReLU(),
            Linear(hidden_units, hidden_units), ReLU(),
            Linear(hidden_units, 2)  # reward and entropy predicted separately
        )

    # noinspection PyMethodOverriding
    def forward(self, obs, action):
        x = (*obs, action)
        return super().forward(x)


class MlpPolicy(Sequential):
    def __init__(self, observation_space, action_space, hidden_units, act_delay=True, obs_delay=True, tbmdp=False):
        super().__init__(
            DelayedMlpModule(observation_space, action_space, is_Q_network=False, act_delay=act_delay, obs_delay=obs_delay, tbmdp=tbmdp), ReLU(),
            Linear(hidden_units, hidden_units), ReLU(),
            TanhNormalLayer(hidden_units, action_space.shape[0])
        )

    # noinspection PyMethodOverriding
    def forward(self, obs):
        return super().forward(obs)


class Mlp(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256, num_critics: int = 2, act_delay: bool = True, obs_delay: bool = True, tbmdp: bool = False):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        self.critics = ModuleList(MlpActionValue(observation_space, action_space, hidden_units, act_delay=act_delay, obs_delay=obs_delay, tbmdp=tbmdp) for _ in range(num_critics))
        self.actor = MlpPolicy(observation_space, action_space, hidden_units, act_delay=act_delay, obs_delay=obs_delay, tbmdp=tbmdp)
        self.critic_output_layers = [c[-1] for c in self.critics]


# # === Testing ==========================================================================================================
#
# if __name__ == "__main__":
#     from tmrl import Training, run
#     from tmrl.util import partial
#     from tmrl.sac import Agent
#
#     Delayed_Sac_Test1 = partial(
#         Training,
#         epochs=2,
#         rounds=10,
#         Agent=partial(Agent, device='cuda', Model=partial(Mlp, act_delay=True, obs_delay=True, tbmdp=True)),
#         Env=partial(RandomDelayEnv, min_observation_delay=0, sup_observation_delay=2, min_action_delay=0, sup_action_delay=2),  # RTRL setting, should get roughly the same behavior as SAC in RTRL
#     )
#
#     Delayed_Sac_Test2 = partial(
#         Training,
#         epochs=2,
#         rounds=10,
#         Agent=partial(Agent, device='cuda', Model=partial(Mlp, act_delay=False, obs_delay=False)),  # random delay information in obs ignored by model
#         Env=partial(RandomDelayEnv, min_observation_delay=0, sup_observation_delay=8, min_action_delay=0, sup_action_delay=2),  # random delays
#     )
#
#     Delayed_Sac_Test3 = partial(
#         Training,
#         epochs=2,
#         rounds=10,
#         Agent=partial(Agent, device='cuda', Model=partial(Mlp, act_delay=True, obs_delay=True)),  # random delay information in obs taken into account by model
#         Env=partial(RandomDelayEnv, min_observation_delay=0, sup_observation_delay=8, min_action_delay=0, sup_action_delay=2),  # random delays
#     )
#
#     Sac_Test = partial(
#         Training,
#         epochs=2,
#         rounds=10,
#         Agent=partial(Agent, device='cuda'),
#         Env=partial(id="Pendulum-v0", real_time=True),
#     )
#
#     # print("--- NOW RUNNING: SAC, normal env, normal MLP model, RTRL setting ---")
#     # run(Sac_Test)
#     print("--- NOW RUNNING: SAC, delayed wrapper, delayed MLP model, RTRL setting ---")
#     run(Delayed_Sac_Test1)
#     print("--- NOW RUNNING: SAC, delayed wrapper, delayed MLP model, random delays setting, ignoring delays in observations ---")
#     run(Delayed_Sac_Test2)
#     print("--- NOW RUNNING: SAC, delayed wrapper, delayed MLP model, random delays setting, taking delays into account in observations ---")
#     run(Delayed_Sac_Test3)
