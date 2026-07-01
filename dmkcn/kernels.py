"""Kernel functions used by the multi-kernel representation learner.

Each kernel maps a scale feature matrix H of shape (N, d) to an (N, N) Gram
matrix K with K[i, j] = kappa(h_i, h_j).

The scDMKC paper uses four kernels, denoted in the ablation table (Table 4) as
G, T, C, P. Per Section 4.5 these are Gaussian, Sigmoid (tanh), Cosine and
Polynomial.  Hyper-parameters of the kernels are NOT given in the paper.

# ASSUMPTION: default hyper-parameters below are standard sklearn-style values
# and are fully configurable so they can be tuned during reproduction.
"""

from __future__ import annotations

import torch


def _pairwise_sq_dists(H: torch.Tensor) -> torch.Tensor:
    """Numerically stable pairwise squared Euclidean distances, shape (N, N)."""
    sq = (H * H).sum(dim=1, keepdim=True)          # (N, 1)
    d2 = sq + sq.t() - 2.0 * (H @ H.t())           # (N, N)
    return d2.clamp_min(0.0)


def cosine_kernel(H: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine (normalised linear) kernel in [-1, 1]."""
    norm = H.norm(dim=1, keepdim=True).clamp_min(eps)
    Hn = H / norm
    return Hn @ Hn.t()


def polynomial_kernel(
    H: torch.Tensor, gamma: float = 1.0, coef0: float = 1.0, degree: int = 3
) -> torch.Tensor:
    """Polynomial kernel: (gamma * <h_i, h_j> + coef0) ** degree."""
    return (gamma * (H @ H.t()) + coef0).pow(degree)


def gaussian_kernel(H: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    """Gaussian / RBF kernel: exp(-gamma * ||h_i - h_j||^2).

    If ``gamma`` is None it defaults to 1 / n_features (sklearn 'scale'-like).
    """
    if gamma is None:
        gamma = 1.0 / H.shape[1]
    return torch.exp(-gamma * _pairwise_sq_dists(H))


def sigmoid_kernel(
    H: torch.Tensor, gamma: float | None = None, coef0: float = 1.0
) -> torch.Tensor:
    """Sigmoid (tanh) kernel: tanh(gamma * <h_i, h_j> + coef0)."""
    if gamma is None:
        gamma = 1.0 / H.shape[1]
    return torch.tanh(gamma * (H @ H.t()) + coef0)


# Registry mapping a short name to (callable, default-kwargs).
# Order matches the paper's "S, C, P, G" listing.
KERNEL_REGISTRY = {
    "sigmoid": sigmoid_kernel,
    "cosine": cosine_kernel,
    "polynomial": polynomial_kernel,
    "gaussian": gaussian_kernel,
}

DEFAULT_KERNELS = ("sigmoid", "cosine", "polynomial", "gaussian")


def build_kernel(H: torch.Tensor, name: str, **kwargs) -> torch.Tensor:
    """Compute a single (N, N) kernel matrix by name."""
    if name not in KERNEL_REGISTRY:
        raise KeyError(f"Unknown kernel '{name}'. Available: {list(KERNEL_REGISTRY)}")
    return KERNEL_REGISTRY[name](H, **kwargs)
