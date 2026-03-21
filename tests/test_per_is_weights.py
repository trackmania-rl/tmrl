"""Tests for Prioritized Experience Replay importance sampling weight normalization.

Verifies that IS weights are normalized by max (not mean/sum) so that the
gradient scaling factor is bounded by 1.0. This is critical for training
stability with PER.
"""

import torch


def _normalize_is_weights_max(raw_weights: torch.Tensor) -> torch.Tensor:
    """Max normalization (correct, used in codebase)."""
    return raw_weights / (raw_weights.max() + 1e-8)


def _normalize_is_weights_mean(raw_weights: torch.Tensor) -> torch.Tensor:
    """Mean normalization (incorrect, would cause gradient explosion)."""
    return raw_weights / (raw_weights.mean() + 1e-8)


class TestISWeightNormalization:
    def test_max_normalization_bounded_by_one(self):
        raw = torch.tensor([0.1, 0.5, 1.0, 10.0, 100.0])
        normed = _normalize_is_weights_max(raw)
        assert normed.max().item() <= 1.0 + 1e-6

    def test_max_normalization_preserves_ratios(self):
        raw = torch.tensor([2.0, 4.0, 8.0])
        normed = _normalize_is_weights_max(raw)
        assert abs(normed[0].item() / normed[1].item() - 0.5) < 1e-5

    def test_mean_normalization_can_exceed_one(self):
        raw = torch.tensor([0.001, 0.001, 0.001, 100.0])
        normed = _normalize_is_weights_mean(raw)
        assert normed.max().item() > 1.0, (
            "Mean normalization allows weights > 1.0, which causes gradient explosion"
        )

    def test_single_element(self):
        raw = torch.tensor([5.0])
        normed = _normalize_is_weights_max(raw)
        assert abs(normed.item() - 1.0) < 1e-5

    def test_uniform_weights(self):
        raw = torch.ones(10) * 3.0
        normed = _normalize_is_weights_max(raw)
        assert torch.allclose(normed, torch.ones(10), atol=1e-5)

    def test_extreme_outlier_capped(self):
        raw = torch.tensor([0.0001, 0.0001, 1000.0])
        normed = _normalize_is_weights_max(raw)
        assert normed.max().item() <= 1.0 + 1e-6
        assert normed[0].item() < 1e-4
