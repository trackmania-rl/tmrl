import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal

from tmrl.actor import TorchActorModule
from tmrl.custom.models.model_blocks import VanillaCNN
from tmrl.custom.models.model_constants import LOG_STD_MAX, LOG_STD_MIN


class SquashedGaussianVanillaCNNActor(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = VanillaCNN(q_net=False)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def forward(self, obs, test=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # NB: this is from Spinning Up (openai/spinningup).
            # FIXME: this formula is mathematically wrong, no idea why it seems to work.
            # Compare with SB3 SquashedGaussian:
            # log_prob -= th.sum(th.log(1 - actions**2 + eps), dim=1)
            # Ref: https://github.com/DLR-RM/stable-baselines3/blob/master/sb3/common/distributions.py
            logp_pi -= (2 * (np.log(2) - pi_action - functional.softplus(-2 * pi_action))).sum(
                axis=1
            )
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


class VanillaCNNQFunction(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)

    def forward(self, obs, act):
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class VanillaCNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # build policy and value functions
        self.actor = SquashedGaussianVanillaCNNActor(observation_space, action_space)
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.cpu().numpy()
