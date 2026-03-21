"""Tests for gSDE (generalized State-Dependent Exploration) noise module.

Verifies:
- reset_noise produces different exploration matrices each call
- Noise is state-dependent (varies with latent input)
- Exploration matrices have correct shapes
"""

import torch
from tmrl.custom.utils.nn import GSDEModule


class TestGSDEModule:
    def setup_method(self):
        self.latent_dim = 16
        self.action_dim = 3
        self.sde = GSDEModule(self.latent_dim, self.action_dim)

    def test_exploration_mat_shape(self):
        assert self.sde.exploration_mat.shape == (self.latent_dim, self.action_dim)

    def test_reset_noise_changes_matrix(self):
        mat_before = self.sde.exploration_mat.clone()
        self.sde.reset_noise(1)
        mat_after = self.sde.exploration_mat
        assert not torch.allclose(mat_before, mat_after), (
            "reset_noise should produce a different exploration matrix"
        )

    def test_batch_exploration_matrices_shape(self):
        batch_size = 8
        self.sde.reset_noise(batch_size)
        assert self.sde.exploration_matrices.shape == (batch_size, self.latent_dim, self.action_dim)

    def test_noise_is_state_dependent(self):
        self.sde.reset_noise(1)
        latent_a = torch.randn(1, self.latent_dim)
        latent_b = torch.randn(1, self.latent_dim)
        noise_a = self.sde.get_noise(latent_a)
        noise_b = self.sde.get_noise(latent_b)
        assert not torch.allclose(noise_a, noise_b), (
            "Different latent states should produce different noise"
        )

    def test_noise_output_shape(self):
        self.sde.reset_noise(1)
        latent = torch.randn(1, self.latent_dim)
        noise = self.sde.get_noise(latent)
        assert noise.shape == (1, self.action_dim)

    def test_get_variance_positive(self):
        latent = torch.randn(4, self.latent_dim)
        var = self.sde.get_variance(latent)
        assert (var >= 0).all()
