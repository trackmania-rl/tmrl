import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal

from tmrl.custom.models.model_blocks import mlp
from tmrl.custom.models.model_constants import LOG_STD_MAX, LOG_STD_MIN
from tmrl.util import prod


def rnn(input_size, rnn_size, rnn_len):
    """
    sizes is ignored for now, expect first values and length
    """
    num_rnn_layers = rnn_len
    assert num_rnn_layers >= 1
    hidden_size = rnn_size

    gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_rnn_layers,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    )
    return gru


class SquashedGaussianRNNActor(nn.Module):
    def __init__(
        self,
        obs_space,
        act_space,
        rnn_size=100,
        rnn_len=2,
        mlp_sizes=(100, 100),
        activation=nn.ReLU,
    ):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        act_limit = act_space.high[0]
        self.rnn = rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size, *list(mlp_sizes)], activation, activation)
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.act_limit = act_limit
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len
        self.dropout = nn.Dropout(0.1)

    def forward(self, obs_seq, test=False, with_logprob=True, save_hidden=False):
        """
        obs: observation
        h: hidden state
        Returns:
            pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        obs_seq_cat = torch.cat(obs_seq, -1)

        net_out, h = self.rnn(obs_seq_cat, h)
        net_out = net_out[:, -1]
        net_out = self.mlp(net_out)
        net_out = self.dropout(net_out)

        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        # Only used for evaluating policy at test time.
        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - functional.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h = h

        return pi_action, logp_pi

    def act(self, obs, test=False):
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)  # artificially add sequence dimension
        with torch.no_grad():
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.cpu().numpy()


class RNNQFunction(nn.Module):
    """
    The action is merged in the latent space after the RNN
    """

    def __init__(
        self,
        obs_space,
        act_space,
        rnn_size=100,
        rnn_len=2,
        mlp_sizes=(100, 100),
        activation=nn.ReLU,
    ):
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = self.rnn(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act, *list(mlp_sizes), 1], activation)
        self.h = None
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len
        self.dropout = nn.Dropout(0.1)

    def forward(self, obs_seq, act, save_hidden=False):
        """
        obs: observation
        h: hidden state
        Returns:
            pi_action, log_pi, h
        """
        self.rnn.flatten_parameters()

        # sequence_len = obs_seq[0].shape[0]
        batch_size = obs_seq[0].shape[0]

        if not save_hidden or self.h is None:
            device = obs_seq[0].device
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=device)
        else:
            h = self.h

        # logging.debug(f"len(obs_seq):{len(obs_seq)}")
        # logging.debug(f"obs_seq[0].shape:{obs_seq[0].shape}")
        # logging.debug(f"obs_seq[1].shape:{obs_seq[1].shape}")
        # logging.debug(f"obs_seq[2].shape:{obs_seq[2].shape}")
        # logging.debug(f"obs_seq[3].shape:{obs_seq[3].shape}")

        obs_seq_cat = torch.cat(obs_seq, -1)

        # logging.debug(f"obs_seq_cat.shape:{obs_seq_cat.shape}")

        net_out, h = self.rnn(obs_seq_cat, h)
        # logging.debug(f"1 net_out.shape:{net_out.shape}")
        net_out = net_out[:, -1]
        # logging.debug(f"2 net_out.shape:{net_out.shape}")
        net_out = torch.cat((net_out, act), -1)
        # logging.debug(f"3 net_out.shape:{net_out.shape}")
        q = self.mlp(net_out)
        q = self.dropout(q)

        if save_hidden:
            self.h = h

        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class RNNActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        rnn_size=100,
        rnn_len=2,
        mlp_sizes=(100, 100),
        activation=nn.ReLU,
    ):
        super().__init__()

        # act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = SquashedGaussianRNNActor(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation
        )
        self.q1 = RNNQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation
        )
        self.q2 = RNNQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation
        )
