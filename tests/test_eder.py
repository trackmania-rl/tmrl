"""Tests for EDER greedy k-DPP diversity-based batch filtering."""

import torch
from tmrl.custom.utils.eder import _rbf_kernel, greedy_kdpp_filter


class TestRbfKernel:
    def test_symmetric(self):
        x = torch.randn(10, 8)
        k_mat = _rbf_kernel(x)
        assert torch.allclose(k_mat, k_mat.T, atol=1e-5)

    def test_diagonal_is_one(self):
        x = torch.randn(10, 8)
        k_mat = _rbf_kernel(x)
        assert torch.allclose(k_mat.diag(), torch.ones(10), atol=1e-5)

    def test_positive_values(self):
        x = torch.randn(10, 8)
        k_mat = _rbf_kernel(x)
        assert (k_mat >= 0).all()

    def test_shape(self):
        x = torch.randn(5, 3)
        k_mat = _rbf_kernel(x)
        assert k_mat.shape == (5, 5)


class TestGreedyKdppFilter:
    def test_returns_k_indices(self):
        features = torch.randn(20, 16)
        keep = greedy_kdpp_filter(features, k=10)
        assert keep.shape == (10,)

    def test_k_greater_than_n_returns_all(self):
        features = torch.randn(5, 8)
        keep = greedy_kdpp_filter(features, k=10)
        assert keep.shape == (5,)

    def test_no_duplicate_indices(self):
        features = torch.randn(30, 16)
        keep = greedy_kdpp_filter(features, k=15)
        assert len(set(keep.tolist())) == 15

    def test_diverse_selection_vs_random(self):
        """EDER-selected features should be more spread out than a random subset."""
        torch.manual_seed(42)
        n, d = 100, 32
        features = torch.randn(n, d)
        keep = greedy_kdpp_filter(features, k=20)

        selected = features[keep]
        random_idx = torch.randperm(n)[:20]
        random_selected = features[random_idx]

        dpp_spread = torch.cdist(selected, selected).mean()
        rnd_spread = torch.cdist(random_selected, random_selected).mean()
        assert dpp_spread >= rnd_spread * 0.8

    def test_indices_in_range(self):
        features = torch.randn(50, 10)
        keep = greedy_kdpp_filter(features, k=25)
        assert (keep >= 0).all()
        assert (keep < 50).all()

    def test_deterministic_with_seed(self):
        torch.manual_seed(0)
        f = torch.randn(30, 8)
        k1 = greedy_kdpp_filter(f.clone(), k=10)
        k2 = greedy_kdpp_filter(f.clone(), k=10)
        assert torch.equal(k1, k2)

    def test_single_selection(self):
        features = torch.randn(10, 5)
        keep = greedy_kdpp_filter(features, k=1)
        assert keep.shape == (1,)
