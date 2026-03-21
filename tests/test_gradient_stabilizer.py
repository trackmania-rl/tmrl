"""Tests for GradientStabilizer EMA-based gradient magnitude stabilisation."""

import torch
import torch.nn as nn
from tmrl.custom.utils.optim import GradientStabilizer


class TestGradientStabilizerBasic:
    def test_returns_grad_norm(self):
        model = nn.Linear(4, 2)
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        gs = GradientStabilizer(ema_decay=0.995)
        norm = gs.step(model.parameters())
        assert norm > 0.0

    def test_no_params_returns_zero(self):
        gs = GradientStabilizer()
        norm = gs.step(iter([]))
        assert norm == 0.0

    def test_ema_initialised_on_first_call(self):
        gs = GradientStabilizer(ema_decay=0.995)
        assert gs._ema is None
        model = nn.Linear(4, 2)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        gs.step(model.parameters())
        assert gs._ema is not None
        assert gs._ema > 0.0

    def test_ema_tracks_gradient_norm(self):
        gs = GradientStabilizer(ema_decay=0.5, warmup=0)
        model = nn.Linear(4, 2)
        norms = []
        for _ in range(10):
            model.zero_grad()
            loss = model(torch.randn(2, 4)).sum()
            loss.backward()
            n = gs.step(model.parameters())
            norms.append(n)
        assert gs.ema_norm > 0
        assert len(norms) == 10


class TestGradientStabilizerClipping:
    def test_large_gradient_is_rescaled(self):
        gs = GradientStabilizer(ema_decay=0.5, warmup=0)
        model = nn.Linear(4, 2)
        for _ in range(20):
            model.zero_grad()
            loss = model(torch.randn(2, 4)).sum()
            loss.backward()
            gs.step(model.parameters())

        ema_before = gs.ema_norm
        assert ema_before > 0

        model.zero_grad()
        big_loss = (model(torch.randn(2, 4)) * 1000).sum()
        big_loss.backward()

        pre_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
        assert pre_norm > ema_before * 5, "Sanity: gradient should be much larger than EMA"

        model.zero_grad()
        big_loss2 = (model(torch.randn(2, 4)) * 1000).sum()
        big_loss2.backward()
        gs.step(model.parameters())

        total_sq = sum(float(p.grad.norm() ** 2) for p in model.parameters() if p.grad is not None)
        post_norm = total_sq**0.5
        assert post_norm < pre_norm, "Post-stabilisation norm should be smaller than raw"

    def test_small_gradient_not_scaled(self):
        """Gradients smaller than EMA should not be rescaled."""
        gs = GradientStabilizer(ema_decay=0.99, warmup=0)
        model = nn.Linear(4, 2)

        for _ in range(20):
            model.zero_grad()
            loss = (model(torch.randn(2, 4)) * 100).sum()
            loss.backward()
            gs.step(model.parameters())

        model.zero_grad()
        small_loss = model(torch.randn(2, 4)).sum() * 0.001
        small_loss.backward()
        grad_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        gs.step(model.parameters())
        grad_after = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        for gb, ga in zip(grad_before, grad_after, strict=False):
            assert torch.allclose(gb, ga, atol=1e-6)


class TestGradientStabilizerWarmup:
    def test_no_scaling_during_warmup(self):
        gs = GradientStabilizer(ema_decay=0.5, warmup=100)
        model = nn.Linear(4, 2)
        for i in range(5):
            model.zero_grad()
            loss = (model(torch.randn(2, 4)) * (1000 if i == 4 else 1)).sum()
            loss.backward()
            grad_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            gs.step(model.parameters())
            grad_after = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            for gb, ga in zip(grad_before, grad_after, strict=False):
                assert torch.allclose(gb, ga, atol=1e-6)
