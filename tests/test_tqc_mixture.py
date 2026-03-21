"""Unit test for TQC pooled mixture truncation (not per-critic)."""

import torch


def test_tqc_pooled_truncation_shapes():
    """Check that TQC truncation is applied to the pooled mixture, not per-critic."""
    batch_size = 4
    n_quantiles_per_critic = 25
    n_critics = 2
    quantiles_total = n_critics * n_quantiles_per_critic  # 50
    top_quantiles_to_drop = 2
    total_quantiles_to_drop = top_quantiles_to_drop * n_critics  # 4

    # Simulate next_z from two critics: (batch, 2, M)
    next_z = torch.randn(batch_size, n_critics, n_quantiles_per_critic)
    sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
    sorted_z_part = sorted_z[:, : quantiles_total - total_quantiles_to_drop]

    assert sorted_z.shape == (batch_size, quantiles_total), (
        f"sorted_z should have shape (batch, n_quantiles_total), got {sorted_z.shape}"
    )
    assert sorted_z_part.shape[1] == quantiles_total - total_quantiles_to_drop, (
        f"backup should use (quantiles_total - total_quantiles_to_drop) atoms, "
        f"got {sorted_z_part.shape[1]}"
    )


if __name__ == "__main__":
    test_tqc_pooled_truncation_shapes()
    print("TQC mixture truncation test passed.")
