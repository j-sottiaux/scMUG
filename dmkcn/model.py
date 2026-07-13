"""scDMKC-like model.

Stabilized implementation of the network described in Yao & Ren (2025),
"Deep multi-kernel cell clustering for single-cell RNA sequencing data".

The model has four parts (Section 3.1):
  1. multi-scale cell representation learning encoder (eq. 1)
  2. multi-kernel representation learner (eqs. 2-4)
  3. ZINB-based multi-kernel representation decoder (eqs. 5-10)
  4. cell clustering module (Student-t soft assignment, eq. 14)

IMPORTANT (full-batch design): the learned cell-cell representation K is an
(N, N) matrix, where N is the number of cells. The ZINB decoder takes K as its
input (D^(0) = K, see eq. 5), so the decoder's input dimension equals N and the
model is trained full-batch. ``n_cells`` must therefore be known at construction
time. Numerical row normalisation means K is not guaranteed to be a symmetric PSD
Gram matrix.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import DEFAULT_KERNELS, build_kernel
from .zinb import ZINBHead


def _mlp_block(in_dim: int, out_dim: int, activation: str) -> nn.Module:
    act = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}[activation]
    return nn.Sequential(nn.Linear(in_dim, out_dim), act())


class MultiScaleEncoder(nn.Module):
    """Encoder returning the representation H^(l) of every hidden layer (eq. 1)."""

    def __init__(self, dims: Sequence[int], activation: str = "relu"):
        super().__init__()
        # dims = [n_genes, 500, 500, 2000, 10]  -> 4 scales H^(1..4)
        self.layers = nn.ModuleList(
            _mlp_block(dims[i], dims[i + 1], activation) for i in range(len(dims) - 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = x
        hs: list[torch.Tensor] = []
        for layer in self.layers:
            h = layer(h)
            hs.append(h)
        return hs  # [H^(1), ..., H^(L)]


class MultiKernelLearner(nn.Module):
    """Two-level adaptive fusion across kernels then across scales (eqs. 2-4).

    # ASSUMPTION: fusion weights are constrained to be non-negative and to sum to
    # one via a softmax (paper says "learnable fusion weight" without specifying
    # the constraint). Following the notation of eq. (2), the kernel weights W_i
    # are shared across scales; the scale weights W^(l) are per scale (eq. 4).

    # ASSUMPTION (numerical): each scale feature H^(l) is L2-normalised per cell
    # before the kernels are computed. The paper does not mention this, but ReLU
    # features are unbounded and the polynomial kernel cubes the inner product, so
    # without it the kernel matrices (and hence L_k) explode by many orders of
    # magnitude, making the unit loss weights lambda=1 meaningless. Normalising the
    # rows bounds every kernel: cosine/sigmoid in [-1, 1], polynomial in [0, 8] for
    # the defaults, gaussian in [exp(-4 gamma), 1]. Toggle with normalize_features.
    """

    def __init__(
        self,
        n_scales: int,
        kernels: Sequence[str] = DEFAULT_KERNELS,
        kernel_kwargs: dict | None = None,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.kernels = list(kernels)
        self.kernel_kwargs = kernel_kwargs or {}
        self.normalize_features = normalize_features
        self.kernel_logits = nn.Parameter(torch.zeros(len(self.kernels)))  # W_i
        self.scale_logits = nn.Parameter(torch.zeros(n_scales))  # W^(l)

    def forward(self, hs: Sequence[torch.Tensor]):
        w_kernel = torch.softmax(self.kernel_logits, dim=0)
        w_scale = torch.softmax(self.scale_logits, dim=0)

        khat_list = []  # fused-over-kernels matrix per scale (the K^(l) variants)
        for h in hs:
            if self.normalize_features:
                h = F.normalize(h, p=2, dim=1, eps=1e-8)
            k_scale = 0.0
            for i, name in enumerate(self.kernels):
                kwargs = self.kernel_kwargs.get(name, {})
                k_i = build_kernel(h, name, **kwargs)  # (N, N)
                k_i = F.normalize(
                    k_i, p=2, dim=1, eps=1e-8
                )  # row-L2 normalisation of each kernel matrix before fusion
                k_scale = k_scale + w_kernel[i] * k_i  # eq. (2)
            khat_list.append(k_scale)

        K = 0.0
        for l, k_scale in enumerate(khat_list):
            K = K + w_scale[l] * k_scale  # eq. (4)
        return K, khat_list


class ZINBDecoder(nn.Module):
    """Decoder reconstructing X' from K, then predicting ZINB params (eqs. 5-8).

    # ASSUMPTION: the paper states D^(0) = K (the (N, N) consistent kernel rep) and
    # D^(L) = X' but does not give the intermediate decoder geometry. We use an MLP
    # from n_cells -> decoder_hidden -> n_genes. Defaults mirror the encoder hidden
    # sizes in reverse.
    """

    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        hidden: Sequence[int] = (2000, 500, 500),
        activation: str = "relu",
    ):
        super().__init__()
        dims = [n_cells, *hidden]
        self.layers = nn.ModuleList(
            _mlp_block(dims[i], dims[i + 1], activation) for i in range(len(dims) - 1)
        )
        self.out = nn.Linear(dims[-1], n_genes)  # X' (preliminary reconstruction)
        self.zinb_head = ZINBHead(n_genes, n_genes)

    def forward(self, K: torch.Tensor):
        d = K
        for layer in self.layers:
            d = layer(d)
        x_prime = self.out(d)  # (N, n_genes)
        pi, mu, theta = self.zinb_head(x_prime)
        return x_prime, pi, mu, theta


class ScDMKC(nn.Module):
    """The complete scDMKC model."""

    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        n_clusters: int,
        encoder_hidden: Sequence[int] = (500, 500, 2000, 10),
        decoder_hidden: Sequence[int] = (2000, 500, 500),
        kernels: Sequence[str] = DEFAULT_KERNELS,
        kernel_kwargs: dict | None = None,
        activation: str = "relu",
        alpha: float = 1.0,
        normalize_kernel_features: bool = True,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_clusters = n_clusters
        self.alpha = alpha

        enc_dims = [n_genes, *encoder_hidden]
        self.encoder = MultiScaleEncoder(enc_dims, activation)
        self.mk_learner = MultiKernelLearner(
            n_scales=len(encoder_hidden),
            kernels=kernels,
            kernel_kwargs=kernel_kwargs,
            normalize_features=normalize_kernel_features,
        )
        self.decoder = ZINBDecoder(n_cells, n_genes, decoder_hidden, activation)

        # Cluster centers live in the row-space of K, i.e. R^N (eq. 14, k_i is a
        # row of K). Initialised later from k-means on K.
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, n_cells))

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor):
        hs = self.encoder(x)  # [H^(1..L)]
        K, khat_list = self.mk_learner(hs)  # (N, N), [K^(l)]
        x_prime, pi, mu, theta = self.decoder(K)
        q = self.soft_assign(K)  # (N, C)
        return {
            "hs": hs,
            "K": K,
            "khat_list": khat_list,
            "x_prime": x_prime,
            "pi": pi,
            "mu": mu,
            "theta": theta,
            "q": q,
        }

    # ------------------------------------------------------- clustering helpers
    def soft_assign(self, K: torch.Tensor) -> torch.Tensor:
        """Student-t soft assignment of each kernel row k_i to centers (eq. 14)."""
        # ||k_i - v_c||^2 : (N, C)
        diff = K.unsqueeze(1) - self.cluster_centers.unsqueeze(0)  # (N, C, N)
        dist2 = diff.pow(2).sum(dim=2)  # (N, C)
        num = (1.0 + dist2 / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        q = num / num.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """Auxiliary target P (eq. 15): p_ic = (q_ic^2 / f_c) / sum_c'(...)."""
        weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(
            1e-12
        )  # f_c = sum_i q_ic
        p = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return p
