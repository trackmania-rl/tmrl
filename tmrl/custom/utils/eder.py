"""Efficient Diversity-based Experience Replay (EDER) via greedy k-DPP.

Implements a greedy approximation to Determinantal Point Process (DPP) sampling
to select the most diverse subset from a batch of feature vectors.

Usage as a secondary batch filter::

    features = encoder(obs)          # (2N, D)
    keep = greedy_kdpp_filter(features, k=N)  # select N most diverse
    obs, actions, rewards = obs[keep], actions[keep], rewards[keep]

Reference:
    Chen & Zhang, "Fast Greedy MAP Inference for Determinantal Point Process
    to Improve Recommendation Diversity", NeurIPS 2018.
"""

from __future__ import annotations

import torch


def _rbf_kernel(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Compute the RBF (Gaussian) kernel matrix for ``x`` of shape ``(N, D)``."""
    sq_dists = torch.cdist(x, x, p=2).pow(2)
    if sigma is None:
        median_dist = sq_dists.median().clamp(min=1e-6)
        sigma = (median_dist * 0.5).sqrt().item()
    return torch.exp(-sq_dists / (2.0 * sigma**2 + 1e-8))


def greedy_kdpp_filter(
    features: torch.Tensor,
    k: int,
    sigma: float | None = None,
) -> torch.Tensor:
    """Select ``k`` most-diverse indices from ``features`` via greedy k-DPP.

    The algorithm greedily picks the index that maximises the log-determinant
    of the selected kernel sub-matrix at each step, using the Cholesky update
    trick for O(N*k^2) total cost (instead of O(N*k^3) from scratch each step).

    Args:
        features: Observation features of shape ``(N, D)``.
        k: Number of diverse indices to select.
        sigma: RBF bandwidth.  ``None`` for median heuristic.

    Returns:
        Long tensor of ``k`` selected indices.
    """
    n = features.shape[0]
    if k >= n:
        return torch.arange(n, device=features.device, dtype=torch.long)

    kernel = _rbf_kernel(features.detach().float(), sigma=sigma)
    diag = kernel.diag().clone()

    selected: list[int] = []
    chol_rows = torch.zeros(k, n, device=features.device, dtype=kernel.dtype)

    for i in range(k):
        # Marginal gain ≈ diag[j] (after deflation)
        best = int(diag.argmax().item())

        selected.append(best)

        if i < k - 1:
            # Update Cholesky factor for the new selection
            if i == 0:
                chol_rows[0] = kernel[best] / (diag[best].sqrt() + 1e-10)
                diag -= chol_rows[0].pow(2)
            else:
                prev = chol_rows[:i, best]
                col = kernel[best] - chol_rows[:i].T @ prev
                denom = diag[best].sqrt() + 1e-10
                chol_rows[i] = col / denom
                diag -= chol_rows[i].pow(2)

            diag.clamp_(min=0.0)
            # Never re-select an already-chosen index
            for s in selected:
                diag[s] = -1.0

    return torch.tensor(selected, device=features.device, dtype=torch.long)
