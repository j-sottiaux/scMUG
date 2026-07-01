"""Loss terms of the ZINB-based self-supervised strategy (eqs. 11-17).

    L = L_r + lambda1 * L_k + lambda2 * L_c + lambda3 * L_ZINB        (eq. 11)

L_ZINB is provided by ``dmkcn.zinb.ZINBLoss``; the other three are here.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def representation_loss(x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
    """L_r = 1/2 * mean((X - X')^2)   (eq. 12, mean form).

    # ASSUMPTION: the paper writes a 1/(2N) * Frobenius-SUM. We use a mean so the
    # term is O(1) and independent of the number of genes, which is required for
    # the paper's unit weights lambda1=lambda2=lambda3=1 to actually balance the
    # four loss terms. Same rationale for kernel_loss below.
    """
    return 0.5 * (x - x_prime).pow(2).mean()


def kernel_loss(
    hs: Sequence[torch.Tensor],
    K: torch.Tensor,
    cluster_centers: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    """L_k = mean_l ||H^(l) - K H^(l)||^2 + ||K - V||^2   (eq. 16, mean form).

    First term: K acts as a self-representation that reconstructs each scale
    feature from the other cells (dimensionally K(N,N) @ H^(l)(N,d) = (N,d)).
    # ASSUMPTION: K is L1-row-normalised inside THIS term only, so K acts as a
    # proper affinity operator and K @ H is a bounded weighted combination of the
    # other cells. Without it K @ H grows ~N-fold and the self-representation term
    # dominates the whole objective (observed k~1850 vs the other terms ~O(1)).
    # The global K (used for clustering and the decoder) is left unchanged.

    Second term, ||K - V||^2:
    # ASSUMPTION: eq. (16) is dimensionally ambiguous because K is (N, N) while
    # "cluster centers V" are (C, N). Following the prose ("bringing each cell's
    # kernel representation closer to the cluster centers"), we read V as the
    # (N, N) matrix whose row i is the center of the cluster cell i belongs to,
    # so the term pulls each row of K toward its assigned center.
    # Mean reduction is used (see representation_loss) so L_k stays O(1).
    """
    K_aff = K / K.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)  # row-normalised
    self_rep = 0.0
    for h in hs:
        self_rep = self_rep + (h - K_aff @ h).pow(2).mean()
    self_rep = self_rep / max(len(hs), 1)

    V = cluster_centers[assignments]            # (N, N): row i = center of cell i
    center_term = (K - V).pow(2).mean()
    return self_rep + center_term


def clustering_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """L_c = KL(P || Q)   (eq. 13)."""
    q = q.clamp_min(1e-12)
    p = p.clamp_min(1e-12)
    return F.kl_div(q.log(), p, reduction="batchmean")
