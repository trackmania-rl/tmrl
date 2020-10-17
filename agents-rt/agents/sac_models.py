from dataclasses import InitVar, dataclass
import torch
import math
import numpy as np
from agents.util import collate, partition
from agents.nn import TanhNormalLayer, SacLinear
from torch.nn import Linear, Sequential, ReLU, ModuleList, Module
import gym


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
    def __init__(self, obs_space, act_space, hidden_units):
        dim_obs = sum(math.prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        super().__init__(
            SacLinear(dim_obs + dim_act, hidden_units), ReLU(),
            SacLinear(hidden_units, hidden_units), ReLU(),
            Linear(hidden_units, 2)
        )

    # noinspection PyMethodOverriding
    def forward(self, obs, action):
        x = torch.cat((*obs, action), -1)
        return super().forward(x)


class MlpPolicy(Sequential):
    def __init__(self, obs_space, act_space, hidden_units=256, act_in_obs=False):
        dim_obs = sum(math.prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        super().__init__(
            SacLinear(dim_obs, hidden_units), ReLU(),
            SacLinear(hidden_units, hidden_units), ReLU(),
            TanhNormalLayer(hidden_units, dim_act)
        )

    # noinspection PyMethodOverriding
    def forward(self, obs):
        return super().forward(torch.cat(obs, -1))  # XXX


class Mlp(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256, num_critics: int = 2, act_in_obs=False):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple), f"{observation_space}"
        self.critics = ModuleList(MlpActionValue(observation_space, action_space, hidden_units) for _ in range(num_critics))
        self.actor = MlpPolicy(observation_space, action_space, hidden_units)
        self.critic_output_layers = [c[-1] for c in self.critics]


# === Testing ==========================================================================================================
class TestMlp(ActorModule):
    def act(self, state, obs, r, done, info, train=False):
        return obs.copy(), state, {}


if __name__ == "__main__":
    pass
