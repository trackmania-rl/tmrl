"""Tests for n-step return computation and bootstrap masking.

Covers:
- 1-step returns fall back to raw rewards
- Multi-step discounting
- Terminal transitions (done=1) cut the bootstrap
- Truncated episodes (done=0) still bootstrap (not tested here directly,
  but the function only receives 'terminated' so truncation keeps bootstrap)
"""

import torch
from tmrl.custom.custom_algorithms._common import _compute_n_step_return_and_bootstrap_mask


class TestNStepReturn:
    def test_1_step_is_identity(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.tensor([0.0, 0.0, 0.0])
        ret, mask = _compute_n_step_return_and_bootstrap_mask(rewards, dones, gamma=0.99, n_steps=1)
        assert torch.allclose(ret.squeeze(), rewards)
        assert (mask.squeeze() == 1.0).all()

    def test_3_step_discounting(self):
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        gamma = 0.9
        ret, _mask = _compute_n_step_return_and_bootstrap_mask(
            rewards, dones, gamma=gamma, n_steps=3
        )
        expected_0 = 1.0 + gamma * 1.0 + gamma**2 * 1.0
        assert abs(ret[0].item() - expected_0) < 1e-5

    def test_done_stops_bootstrap(self):
        rewards = torch.tensor([1.0, 1.0, 1.0])
        dones = torch.tensor([0.0, 1.0, 0.0])
        gamma = 0.99
        _ret, mask = _compute_n_step_return_and_bootstrap_mask(
            rewards, dones, gamma=gamma, n_steps=3
        )
        assert mask[0].item() == 0.0, (
            "Bootstrap should be 0 when done=1 occurs within the n-step window"
        )

    def test_no_done_bootstraps(self):
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        dones = torch.tensor([0.0, 0.0, 0.0, 0.0])
        _, mask = _compute_n_step_return_and_bootstrap_mask(rewards, dones, gamma=0.99, n_steps=3)
        assert mask[0].item() == 1.0, "Should bootstrap when no terminal within n-step window"

    def test_output_shapes(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dones = torch.tensor([0.0, 0.0, 1.0, 0.0])
        ret, mask = _compute_n_step_return_and_bootstrap_mask(rewards, dones, gamma=0.99, n_steps=2)
        assert ret.shape == (4, 1)
        assert mask.shape == (4, 1)

    def test_truncation_keeps_bootstrap(self):
        """When only 'terminated' is used (not 'truncated'), done=0 means bootstrap."""
        rewards = torch.tensor([1.0, 1.0])
        dones_terminated = torch.tensor([0.0, 0.0])
        _, mask = _compute_n_step_return_and_bootstrap_mask(
            rewards, dones_terminated, gamma=0.99, n_steps=2
        )
        assert mask[0].item() == 1.0, (
            "Truncated episodes pass done=0 so bootstrap must be preserved"
        )
