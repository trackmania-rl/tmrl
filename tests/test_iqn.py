"""Tests for IQN Q-network, DQN actor, and quantile Huber loss."""

import numpy as np
import torch
from gymnasium import spaces
from tmrl.custom.custom_algorithms.iqn import _quantile_huber_loss
from tmrl.custom.models.DQNNet import (
    CosineEmbedding,
    DQNActor,
    DuelingHead,
    IQNQNetwork,
)


class TestCosineEmbedding:
    def test_output_shape(self):
        embed = CosineEmbedding(n_cos=32, embed_dim=64)
        tau = torch.rand(4, 8)
        out = embed(tau)
        assert out.shape == (4, 8, 64)

    def test_output_range(self):
        """ReLU output should be non-negative."""
        embed = CosineEmbedding(n_cos=64, embed_dim=128)
        tau = torch.rand(2, 16)
        out = embed(tau)
        assert (out >= 0).all()


class TestDuelingHead:
    def test_output_shape(self):
        head = DuelingHead(hidden_dim=64, n_actions=10)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 10)

    def test_advantage_centering(self):
        """Mean advantage across actions should be near zero."""
        head = DuelingHead(hidden_dim=64, n_actions=10)
        x = torch.randn(8, 64)
        q = head(x)
        v = head.value_stream(x)
        a = head.advantage_stream(x)
        reconstructed = v + a - a.mean(dim=-1, keepdim=True)
        assert torch.allclose(q, reconstructed, atol=1e-5)


class TestIQNQNetwork:
    def setup_method(self):
        self.obs_space = (
            spaces.Box(-1, 1, shape=(60,)),
            spaces.Box(-1, 1, shape=(3,)),
        )
        self.n_actions = 78

    def test_forward_shapes(self):
        net = IQNQNetwork(self.obs_space, n_actions=self.n_actions, hidden_dim=64, num_blocks=1)
        obs = (torch.randn(2, 60), torch.randn(2, 3))
        qv, tau = net(obs, n_quantiles=16)
        assert qv.shape == (2, 16, self.n_actions)
        assert tau.shape == (2, 16)

    def test_q_values_shape(self):
        net = IQNQNetwork(self.obs_space, n_actions=self.n_actions, hidden_dim=64, num_blocks=1)
        obs = (torch.randn(3, 60), torch.randn(3, 3))
        q = net.q_values(obs, n_quantiles=8)
        assert q.shape == (3, self.n_actions)

    def test_explicit_tau(self):
        net = IQNQNetwork(self.obs_space, n_actions=self.n_actions, hidden_dim=64, num_blocks=1)
        obs = (torch.randn(2, 60), torch.randn(2, 3))
        tau_in = torch.tensor([[0.25, 0.75], [0.1, 0.9]])
        qv, tau_out = net(obs, tau=tau_in)
        assert torch.equal(tau_in, tau_out)
        assert qv.shape == (2, 2, self.n_actions)


class TestDQNActor:
    def setup_method(self):
        self.obs_space = (
            spaces.Box(-1, 1, shape=(60,)),
            spaces.Box(-1, 1, shape=(3,)),
        )
        self.act_space = spaces.Discrete(78)

    def test_act_returns_integer(self):
        actor = DQNActor(
            self.obs_space, self.act_space, hidden_dim=64, num_blocks=1, n_actions=78, epsilon=0.0
        )
        obs = (torch.randn(1, 60), torch.randn(1, 3))
        action = actor.act(obs, test=True)
        assert action.dtype == np.int64

    def test_greedy_action_in_range(self):
        actor = DQNActor(
            self.obs_space, self.act_space, hidden_dim=64, num_blocks=1, n_actions=78, epsilon=0.0
        )
        obs = (torch.randn(1, 60), torch.randn(1, 3))
        for _ in range(10):
            action = actor.act(obs, test=True)
            assert 0 <= int(action) < 78

    def test_epsilon_one_explores(self):
        """With epsilon=1.0, all actions should be random (uniform)."""
        actor = DQNActor(
            self.obs_space, self.act_space, hidden_dim=64, num_blocks=1, n_actions=78, epsilon=1.0
        )
        obs = (torch.randn(1, 60), torch.randn(1, 3))
        actions = {int(actor.act(obs, test=False)) for _ in range(200)}
        assert len(actions) > 1


class TestQuantileHuberLoss:
    def test_zero_loss_constant_quantiles(self):
        """When all quantile values are the same constant, pairwise deltas are zero."""
        vals = torch.ones(4, 8) * 3.0
        tau = torch.rand(4, 8)
        loss = _quantile_huber_loss(vals, vals, tau)
        assert loss.item() < 1e-5

    def test_loss_is_scalar(self):
        current = torch.randn(4, 8)
        target = torch.randn(4, 8)
        tau = torch.rand(4, 8)
        loss = _quantile_huber_loss(current, target, tau)
        assert loss.dim() == 0

    def test_loss_positive_for_different_quantiles(self):
        current = torch.zeros(4, 8)
        target = torch.ones(4, 8)
        tau = torch.rand(4, 8)
        loss = _quantile_huber_loss(current, target, tau)
        assert loss.item() > 0
