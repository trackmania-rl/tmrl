"""
InfoNCE (contrastive) loss for goal-conditioned / contrastive RL.

Enables training a critic f(s,a,g) = -||phi(s,a) - psi(g)||_2 with InfoNCE,
so the policy can be trained to maximize f (e.g. for goal-reaching or auxiliary loss).
Paper: Contrastive RL (Eysenbach et al.); scaling depth (2503.14858).
"""

import torch
from torch import nn


class StateActionGoalEncoders(nn.Module):
    """
    Encoders phi(s,a) and psi(g) for contrastive learning.
    f(s,a,g) = -||phi(s,a) - psi(g)||_2 (L2 distance; lower = better match).
    """

    def __init__(self, sa_dim: int, g_dim: int, hidden_dim: int = 256, out_dim: int = 64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(sa_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.psi = nn.Sequential(
            nn.Linear(g_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward_sa(self, s, a):
        """Concatenate s,a and return phi(s,a)."""
        if isinstance(s, (list, tuple)):
            sa = torch.cat([*s, a], dim=-1)
        else:
            sa = torch.cat([s, a], dim=-1)
        return self.phi(sa)

    def forward_g(self, g):
        if isinstance(g, (list, tuple)):
            g = torch.cat(g, dim=-1)
        return self.psi(g)

    def f(self, s, a, g):
        """Critic f(s,a,g) = -||phi(s,a) - psi(g)||_2 (higher = closer to goal)."""
        phi_sa = self.forward_sa(s, a)
        psi_g = self.forward_g(g)
        return -torch.norm(phi_sa - psi_g, dim=-1, p=2)


def info_nce_loss(
    phi_sa: torch.Tensor,
    psi_g_pos: torch.Tensor,
    psi_g_neg: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    InfoNCE loss: -log( exp(f_pos/tau) / (exp(f_pos/tau) + sum_j exp(f_neg_j/tau)) ).

    phi_sa: (B, D) state-action embeddings.
    psi_g_pos: (B, D) positive goal embeddings (e.g. future state from same trajectory).
    psi_g_neg: (B, K, D) negative goal embeddings (e.g. random goals). First dim must match B.
    Returns scalar loss.
    """
    b = phi_sa.shape[0]
    if psi_g_neg.shape[0] != b:
        raise ValueError(
            f"psi_g_neg first dim ({psi_g_neg.shape[0]}) must match batch size ({b}) from phi_sa."
        )
    if psi_g_neg.dim() == 3:
        k = psi_g_neg.shape[1]
        phi_sa = phi_sa.unsqueeze(1)
        f_pos = -torch.norm(phi_sa - psi_g_pos.unsqueeze(1), dim=-1, p=2).squeeze(1) / temperature
        f_neg = -torch.norm(phi_sa.expand(-1, k, -1) - psi_g_neg, dim=-1, p=2) / temperature
        logits = torch.cat([f_pos.unsqueeze(1), f_neg], dim=1)
    else:
        # 2D (B, D): one negative per batch item
        f_pos = -torch.norm(phi_sa - psi_g_pos, dim=-1, p=2) / temperature
        f_neg = -torch.norm(phi_sa - psi_g_neg, dim=-1, p=2) / temperature
        logits = torch.stack([f_pos, f_neg], dim=1)
    labels = torch.zeros(b, dtype=torch.long, device=phi_sa.device)
    return nn.functional.cross_entropy(logits, labels)


def info_nce_loss_from_encoders(
    encoder: StateActionGoalEncoders,
    s: torch.Tensor,
    a: torch.Tensor,
    g_pos: torch.Tensor,
    g_neg: torch.Tensor,
    temperature: float = 1.0,
    num_negatives: int | None = None,
) -> torch.Tensor:
    """
    Compute InfoNCE loss using StateActionGoalEncoders.

    g_neg: (B, K, g_dim), or (B, g_dim) for K=1, or (B*K, g_dim) if num_negatives=K.
    When g_neg is 2D, first dim must be B (batch size) or B*K; if B*K, num_negatives=K.
    """
    phi_sa = encoder.forward_sa(s, a)
    b = phi_sa.shape[0]
    psi_g_pos = encoder.forward_g(g_pos)
    if isinstance(g_neg, (list, tuple)):
        g_neg = torch.stack(g_neg, dim=1)
    if g_neg.dim() == 2:
        m, g_dim = g_neg.shape[0], g_neg.shape[1]
        if m == b:
            # (B, g_dim): one negative per batch item -> (B, 1, D)
            psi_g_neg = encoder.forward_g(g_neg)
            psi_g_neg = psi_g_neg.unsqueeze(1)
        elif num_negatives is not None and m == b * num_negatives:
            # (B*K, g_dim): reshape to (B, K, g_dim)
            k = num_negatives
            g_neg = g_neg.view(b, k, g_dim)
            psi_g_neg = encoder.forward_g(g_neg.view(b * k, -1)).view(b, k, -1)
        else:
            raise ValueError(
                f"g_neg has shape (M={m}, g_dim). M must equal B={b}, "
                f"or B*K with num_negatives passed. Got num_negatives={num_negatives}."
            )
    else:
        b_neg, k, _ = g_neg.shape
        if b_neg != b:
            raise ValueError(
                f"g_neg first dimension ({b_neg}) must match batch size ({b}) from (s, a)."
            )
        psi_g_neg = encoder.forward_g(g_neg.view(b * k, -1)).view(b, k, -1)
    return info_nce_loss(phi_sa, psi_g_pos, psi_g_neg, temperature)
