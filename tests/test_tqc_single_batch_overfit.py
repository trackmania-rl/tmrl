"""Single-batch overfit test for TQC: on a fixed batch, losses should converge (no explosion).

If the algorithm is correct, training on one static batch should drive critic and actor
losses toward convergence. Exploding or oscillating losses indicate a bug in the
loss, target computation, or optimizer.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class _QuantileCritic(nn.Module):
    """Minimal quantile critic matching the interface expected by TQCAgent."""

    def __init__(self, obs_dim, act_dim, num_quantiles=5, hidden=64):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_quantiles),
        )

    def forward(self, obs, act):
        if isinstance(obs, (tuple, list)):
            obs = torch.cat(obs, dim=-1)
        return self.net(torch.cat([obs, act], dim=-1))


class _SimpleActor(nn.Module):
    """Minimal squashed-Gaussian actor supporting return_pre_tanh_mean for TQCAgent."""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.mu_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
        self.act_limit = 1.0

    def forward(self, obs, test=False, with_logprob=True, return_pre_tanh_mean=False):
        if isinstance(obs, (tuple, list)):
            obs = torch.cat(obs, dim=-1)
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h).clamp(-20.0, 2.0)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        pi_action = mu if test else dist.rsample()
        if with_logprob:
            logp = dist.log_prob(pi_action).sum(dim=-1)
            logp -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(dim=-1)
        else:
            logp = None
        squashed = torch.tanh(pi_action) * self.act_limit
        if return_pre_tanh_mean:
            return squashed, logp, mu
        return squashed, logp

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test=test, with_logprob=False)
            return a.cpu().numpy()


_NUM_QUANTILES = 5


class _TestTQCModel(nn.Module):
    """Minimal TQC-compatible actor-critic for isolated training tests."""

    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = action_space.shape[0]
        self.actor = _SimpleActor(obs_dim, act_dim)
        self.q1 = _QuantileCritic(obs_dim, act_dim, _NUM_QUANTILES)
        self.q2 = _QuantileCritic(obs_dim, act_dim, _NUM_QUANTILES)


def test_tqc_single_batch_overfit_no_explosion():
    """Run TQC train on a single fixed batch for many steps; critic must overfit."""
    try:
        from gymnasium import spaces
        from tmrl.custom.custom_algorithms.tqc import TQCAgent
    except ImportError:
        pytest.skip("tmrl or gymnasium not available")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    agent = TQCAgent(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        model_cls=_TestTQCModel,
        n_steps=1,
        learn_entropy_coef=False,
        alpha=0.01,
        top_quantiles_to_drop=1,
        quantiles_number=_NUM_QUANTILES,
    )

    batch_size = 32
    o = torch.randn(batch_size, 10, device=device)
    a = torch.randn(batch_size, 3, device=device).clamp(-1, 1)
    r = torch.ones(batch_size, device=device)
    o2 = torch.randn(batch_size, 10, device=device)
    d = torch.zeros(batch_size, device=device)

    batch = (o, a, r, o2, d)

    initial_q_loss = float("inf")

    for i in range(500):
        stats = agent.train(batch, epoch=0, batch_index=i, iters=500)
        if i == 0:
            initial_q_loss = stats["losses/critic"]

    final_q_loss = stats["losses/critic"]

    assert final_q_loss < initial_q_loss, "Critic loss did not decrease on a fixed batch!"
    assert final_q_loss < 0.1, (
        f"Critic failed to overfit to static batch. Final loss: {final_q_loss}"
    )
    assert not np.isnan(final_q_loss), "Loss exploded to NaN."


if __name__ == "__main__":
    test_tqc_single_batch_overfit_no_explosion()
