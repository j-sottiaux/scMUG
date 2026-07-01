"""Zero-Inflated Negative Binomial (ZINB) reconstruction loss.

Implements equations (9)-(10) of scDMKC, in the standard numerically stable form
used by scDeepCluster / scziDesk (which scDMKC builds on).

# ASSUMPTION: the ZINB likelihood is evaluated on RAW counts, with the NB mean
# scaled by a per-cell size factor. The paper is not explicit about this, but it
# is the universal convention for ZINB-based scRNA-seq autoencoders and is
# required for the count model to be well-posed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZINBLoss(nn.Module):
    """Negative log-likelihood of the ZINB distribution (mean over cells/genes)."""

    def __init__(self, ridge_lambda: float = 0.0, eps: float = 1e-10):
        super().__init__()
        self.ridge_lambda = ridge_lambda
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,            # raw counts (N, G)
        mu: torch.Tensor,           # NB mean (N, G), strictly positive
        theta: torch.Tensor,        # dispersion (N, G), strictly positive
        pi: torch.Tensor,           # dropout / zero-inflation prob (N, G) in (0, 1)
        scale_factor: torch.Tensor | None = None,  # (N, 1) size factors
    ) -> torch.Tensor:
        eps = self.eps
        if scale_factor is not None:
            mu = mu * scale_factor

        theta = theta.clamp_max(1e6)

        # Negative binomial log-likelihood.
        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(x + 1.0)
            - torch.lgamma(x + theta + eps)
        )
        t2 = (theta + x) * torch.log1p(mu / (theta + eps)) + (
            x * (torch.log(theta + eps) - torch.log(mu + eps))
        )
        nb_case = t1 + t2 - torch.log(1.0 - pi + eps)

        # Zero-inflation case (x == 0).
        zero_nb = torch.pow(theta / (theta + mu + eps), theta)
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)

        result = torch.where(x < 1e-8, zero_case, nb_case)

        if self.ridge_lambda > 0.0:
            result = result + self.ridge_lambda * pi.pow(2)

        return result.mean()


class ZINBHead(nn.Module):
    """Three parallel FC layers predicting (pi, mu, theta) from X' (eqs 6-8)."""

    def __init__(self, in_dim: int, n_genes: int):
        super().__init__()
        self.pi = nn.Linear(in_dim, n_genes)     # -> sigmoid
        self.mu = nn.Linear(in_dim, n_genes)     # -> exp
        self.theta = nn.Linear(in_dim, n_genes)  # -> exp

    def forward(self, x_prime: torch.Tensor):
        pi = torch.sigmoid(self.pi(x_prime))
        # clamp before exp for numerical stability (MeanAct / DispAct in scDeepCluster)
        mu = torch.clamp(torch.exp(self.mu(x_prime)), min=1e-5, max=1e6)
        theta = torch.clamp(F.softplus(self.theta(x_prime)), min=1e-4, max=1e4)
        return pi, mu, theta
