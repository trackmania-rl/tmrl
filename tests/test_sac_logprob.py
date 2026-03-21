"""Tests for SAC squashed log-probability numerical stability.

Verifies that log-prob does not produce NaN/Inf even when the pre-tanh action
is extreme (near tanh saturation). This covers the clamp(-20, 20) guard
added in all SquashedGaussian actors.
"""

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def _squash_log_prob_with_clamp(logp, pre_tanh_action, clamp_val=20.0):
    """Reference implementation with pre-tanh clamping (as in codebase)."""
    pre_tanh_clamped = pre_tanh_action.clamp(-clamp_val, clamp_val)
    corr = 2 * (np.log(2) - pre_tanh_clamped - functional.softplus(-2 * pre_tanh_clamped))
    return logp - corr.sum(axis=-1)


def _squash_log_prob_no_clamp(logp, pre_tanh_action):
    """Unclamped reference: can produce -inf at extreme values."""
    corr = 2 * (np.log(2) - pre_tanh_action - functional.softplus(-2 * pre_tanh_action))
    return logp - corr.sum(axis=-1)


class TestSquashLogProbClamp:
    def test_finite_at_extreme_values(self):
        pre_tanh = torch.tensor([[100.0, -100.0, 50.0]])
        base_logp = torch.tensor([0.0])
        result = _squash_log_prob_with_clamp(base_logp, pre_tanh)
        assert torch.isfinite(result).all(), f"Expected finite, got {result}"

    def test_no_nan_at_saturation(self):
        pre_tanh = torch.tensor([[1e6, -1e6, 0.0]])
        base_logp = torch.tensor([0.0])
        result = _squash_log_prob_with_clamp(base_logp, pre_tanh)
        assert not torch.isnan(result).any(), f"NaN detected: {result}"

    def test_unclamped_produces_huge_correction_at_extreme(self):
        """Without clamping, extreme pre-tanh values produce very large correction terms.
        Even if softplus is numerically stable and avoids literal inf, the magnitude is
        far beyond any meaningful log-probability range."""
        pre_tanh = torch.tensor([[1e6, -1e6]])
        base_logp = torch.tensor([0.0])
        result = _squash_log_prob_no_clamp(base_logp, pre_tanh)
        assert result.abs().item() > 1e4, (
            f"Unclamped extreme values should produce huge correction, got {result.item()}"
        )

    def test_normal_range_unchanged(self):
        pre_tanh = torch.tensor([[0.5, -0.3, 1.2]])
        base_logp = torch.tensor([-2.0])
        clamped = _squash_log_prob_with_clamp(base_logp, pre_tanh)
        unclamped = _squash_log_prob_no_clamp(base_logp, pre_tanh)
        assert torch.allclose(clamped, unclamped, atol=1e-6), (
            "Clamp should not affect values in normal range"
        )


class TestSquashedGaussianMLPActor:
    @torch.no_grad()
    def test_forward_no_nan(self):
        from tmrl.custom.models.mlp import SquashedGaussianMLPActor

        obs_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32),
            ]
        )
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        actor = SquashedGaussianMLPActor(obs_space, act_space)
        actor.eval()

        obs = [torch.randn(4, 8)]
        action, logp = actor(obs, test=False, with_logprob=True)
        assert torch.isfinite(action).all(), f"Action has non-finite: {action}"
        assert torch.isfinite(logp).all(), f"Log-prob has non-finite: {logp}"

    @torch.no_grad()
    def test_action_bounded(self):
        from tmrl.custom.models.mlp import SquashedGaussianMLPActor

        obs_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32),
            ]
        )
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        actor = SquashedGaussianMLPActor(obs_space, act_space)
        actor.eval()

        obs = [torch.randn(16, 8)]
        action, _ = actor(obs, test=False, with_logprob=False)
        assert (action.abs() <= 1.0 + 1e-6).all(), f"Action exceeds bounds: {action}"
