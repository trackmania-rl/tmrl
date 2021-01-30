import gym
import torch
from torch.nn import Linear, LSTM, GRU
import numpy as np

from agents.nn import TanhNormalLayer
from agents.sac_models import ActorModule
from agents.util import collate, partition


class StatefulActorModule(ActorModule):
    def act(self, state, obs, r, done, info, train=False):
        """actually uses the state and updates it within the actor"""
        state = self.reset() if state is None else state
        state, obs = collate([(state, obs)], device=self.device)
        with torch.no_grad():
            action_distribution, next_state = self.actor(state, obs)
            action = action_distribution.sample() if train else action_distribution.sample_deterministic()
        (action, next_state), = partition((action, next_state))
        return action, next_state, []

    def reset(self):
        raise NotImplementedError()


class GruModel(StatefulActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        input_dim = sum(s.shape[0] for s in observation_space)
        self.gru = GRU(input_dim, hidden_units)

        self.hidden_units = hidden_units
        self.critic_layer = Linear(hidden_units, 1)
        self.actor_layer = TanhNormalLayer(hidden_units, action_space.shape[0])
        self.critic_output_layers = (self.critic_layer,)

    def actor(self, memory_state, x):
        a, memory_state, _, _ = self(memory_state, x)
        return a, memory_state

    def reset(self):
        return np.random.standard_normal(self.hidden_units).astype(np.float32)

    def forward(self, memory_state, x):
        # self.lstm.flatten_parameters()
        assert isinstance(x, tuple)
        x = torch.cat(x, dim=1)
        batchsize = x.shape[0]
        (hidden_activations,), ((hn,), (cn,)) = self.gru(x[None], memory_state[None])
        v = self.critic_layer(hidden_activations)
        action_distribution = self.actor_layer(hidden_activations)
        return action_distribution, (hn, cn), (v,), (hidden_activations,)


class LstmModel(StatefulActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        input_dim = sum(s.shape[0] for s in observation_space)
        self.lstm = LSTM(input_dim, hidden_units)

        # initialize such that recurrence is suppressed (direct input-output computation dominates)
        with torch.no_grad():
            self.lstm.bias_ih_l0[0:hidden_units // 4].fill_(4.)  # ii
            self.lstm.bias_hh_l0[0:hidden_units // 4].fill_(4.)  # hi
            self.lstm.bias_ih_l0[-hidden_units // 4:].fill_(4.)  # io
            self.lstm.bias_hh_l0[-hidden_units // 4:].fill_(4.)  # ho
            self.lstm.bias_ih_l0[hidden_units // 4:hidden_units // 2].fill_(-4.)  # if
            self.lstm.bias_hh_l0[hidden_units // 4:hidden_units // 2].fill_(-4.)  # hf

        self.hidden_units = hidden_units
        self.critic_layer = Linear(hidden_units, 1)
        self.actor_layer = TanhNormalLayer(hidden_units, action_space.shape[0])
        self.critic_output_layers = (self.critic_layer,)

    def actor(self, memory_state, x):
        a, memory_state, _, _ = self(memory_state, x)
        return a, memory_state

    def reset(self):
        return (np.random.standard_normal(self.hidden_units).astype(np.float32),
                np.random.standard_normal(self.hidden_units).astype(np.float32))

    def forward(self, memory_state, x):
        self.lstm.flatten_parameters()
        assert isinstance(x, tuple)
        x = torch.cat(x, dim=1)
        batchsize = x.shape[0]
        hn, cn = memory_state
        (hidden_activations,), ((hn,), (cn,)) = self.lstm(x[None], (hn[None], cn[None]))
        v = self.critic_layer(hidden_activations)
        action_distribution = self.actor_layer(hidden_activations)
        return action_distribution, (hn, cn), (v,), (hidden_activations,)


class DoubleActorModule(StatefulActorModule):
    @property
    def critic_output_layers(self):
        return self.a.critic_output_layers + self.b.critic_output_layers

    def actor(self, state, x):
        action_distribution, next_state, *_ = self.a(state[0], x)
        return action_distribution, (next_state, state[1])

    def reset(self):
        return self.a.reset(), self.b.reset()

    def forward(self, state, x):
        action_distribution, state_0, v0, h0 = self.a(state[0], x)
        _, state_1, v1, h1 = self.b(state[1], x)
        return action_distribution, (state_0, state_1), v0 + v1, h0 + h1  # note that the + here is not addition but tuple concatenation!


class LstmDouble(DoubleActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256):
        super().__init__()
        self.a = LstmModel(observation_space, action_space, hidden_units=hidden_units)
        self.b = LstmModel(observation_space, action_space, hidden_units=hidden_units)
