"""Tests for SimbaV2 architecture: hyperspherical normalization blocks."""

import torch
from tmrl.custom.models.model_blocks import (
    HypersphericalLinear,
    SimbaV2Backbone,
    SimbaV2Block,
    _l2_normalize,
    simba_v2_backbone,
)


class TestL2Normalize:
    def test_unit_norm(self):
        x = torch.randn(4, 16)
        normed = _l2_normalize(x, dim=-1)
        norms = normed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_zero_vector_safe(self):
        x = torch.zeros(2, 8)
        normed = _l2_normalize(x, dim=-1)
        assert torch.isfinite(normed).all()


class TestHypersphericalLinear:
    def test_output_shape(self):
        layer = HypersphericalLinear(16, 32)
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_weights_unit_norm_after_init(self):
        layer = HypersphericalLinear(16, 32)
        row_norms = layer.weight.data.norm(dim=1)
        assert torch.allclose(row_norms, torch.ones(32), atol=1e-5)

    def test_project_weights_renormalizes(self):
        layer = HypersphericalLinear(16, 32)
        layer.weight.data += torch.randn_like(layer.weight.data) * 0.5
        layer.project_weights()
        row_norms = layer.weight.data.norm(dim=1)
        assert torch.allclose(row_norms, torch.ones(32), atol=1e-5)

    def test_scaler_learnable(self):
        layer = HypersphericalLinear(16, 32)
        assert layer.scaler.requires_grad


class TestSimbaV2Block:
    def test_output_shape(self):
        block = SimbaV2Block(dim=64)
        h = _l2_normalize(torch.randn(4, 64))
        out = block(h)
        assert out.shape == (4, 64)

    def test_output_unit_norm(self):
        block = SimbaV2Block(dim=64)
        h = _l2_normalize(torch.randn(4, 64))
        out = block(h)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-4)

    def test_lerp_alpha_bounded(self):
        """Alpha after sigmoid should be in (0, 1)."""
        block = SimbaV2Block(dim=32)
        alpha = block.alpha.sigmoid()
        assert (alpha > 0).all()
        assert (alpha < 1).all()


class TestSimbaV2Backbone:
    def test_output_shape(self):
        bb = SimbaV2Backbone(input_dim=20, hidden_dim=64, num_blocks=2)
        x = torch.randn(4, 20)
        out = bb(x)
        assert out.shape == (4, 64)

    def test_output_unit_norm(self):
        bb = SimbaV2Backbone(input_dim=20, hidden_dim=64, num_blocks=2)
        x = torch.randn(4, 20)
        out = bb(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-4)

    def test_project_weights_idempotent(self):
        bb = SimbaV2Backbone(input_dim=20, hidden_dim=64, num_blocks=2)
        bb.project_weights()
        row_norms = bb.input_proj.weight.data.norm(dim=1)
        assert torch.allclose(row_norms, torch.ones(row_norms.shape[0]), atol=1e-5)

    def test_factory_function(self):
        bb = simba_v2_backbone(20, 64, 3)
        assert isinstance(bb, SimbaV2Backbone)
        x = torch.randn(2, 20)
        out = bb(x)
        assert out.shape == (2, 64)

    def test_gradient_flows(self):
        bb = SimbaV2Backbone(input_dim=10, hidden_dim=32, num_blocks=2)
        x = torch.randn(2, 10, requires_grad=True)
        out = bb(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
