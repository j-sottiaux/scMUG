"""Integration adapter: use scDMKC as scMUG's per-GFM representation generator.

Drop-in replacement for scMUG's ZINB-autoencoder block B. For one gene functional
module (GFM), it trains scDMKC on that module's genes, takes the normalized
cell-cell representation K, and returns a spectral embedding of K that scMUG's
block C consumes exactly like the original 32-D autoencoder latent.

Usage in scMUG.py -- the ~8 block-B lines become:

    from dmkcn.integration import dmkcn_block_b
    ...
    latent = dmkcn_block_b(adata, gene_list, n_clusters=cluster_number,
                           d=32, seed=seed)
    latent = latent.reshape((latent.shape[0], 1, -1))

Everything else in scMUG (reshape, concat, block C, block D) is unchanged.

Two things to keep in mind
--------------------------
1. Why a spectral embedding of K, not scDMKC's 10-D bottleneck: the bottleneck is
   trained only for reconstruction and as kernel *input*; the discriminative
   structure lives in K. We turn K into a low-dim embedding that carries that
   structure, so block C works unchanged.

2. ZINB target confound: this adapter feeds integer counts (adata.raw.X) as scDMKC
   intends. Feeding scaled/z-scored values to a ZINB likelihood is invalid when
   they are negative, so zinb_on_counts=False is guarded and should be reserved for
   explicit non-negative target ablations.
"""

from __future__ import annotations

import numpy as np
from sklearn.manifold import SpectralEmbedding

from .preprocessing import from_scmug_anndata
from .trainer import ScDMKCTrainer


def affinity_from_K(K: np.ndarray, nonneg: str = "clip") -> np.ndarray:
    """Symmetric, non-negative affinity from the (possibly asymmetric, signed) K.

    After the per-kernel row-L2 normalisation, K is neither symmetric nor
    non-negative, whereas a spectral-embedding affinity must be both.
      - symmetrise: (K + K^T) / 2
      - non-negativity:
          'clip'  -> max(S, 0): anti-similarities (negative cosine/sigmoid) map to
                     zero affinity. Most faithful to "affinity = similarity".
          'abs'   -> |S|: treats strong anti-similarity as strong affinity (rarely
                     what you want, kept for comparison).
          'shift' -> S - min(S): preserves ordering, makes everything positive.
    """
    K = np.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square 2D matrix, got shape {K.shape}.")
    if K.shape[0] < 2:
        raise ValueError("K must contain at least two cells.")
    if not np.isfinite(K).all():
        raise ValueError("K contains NaN or infinite values.")

    S = (K + K.T) / 2.0
    if nonneg == "clip":
        A = np.clip(S, 0.0, None)
    elif nonneg == "abs":
        A = np.abs(S)
    elif nonneg == "shift":
        A = S - S.min()
    else:
        raise ValueError(f"unknown nonneg='{nonneg}'")
    if not np.isfinite(A).all():
        raise ValueError("Affinity derived from K contains NaN or infinite values.")
    if not np.any(A > 0):
        raise ValueError(
            "Affinity derived from K is all zero; spectral embedding is undefined."
        )
    return A.astype(np.float64)


def spectral_embedding_from_K(
    K: np.ndarray, d: int = 32, seed: int = 0, nonneg: str = "clip"
) -> np.ndarray:
    """Return a (n_cells, d) spectral embedding of the kernel representation K."""
    if d <= 0:
        raise ValueError(f"d must be strictly positive, got {d}.")
    A = affinity_from_K(K, nonneg=nonneg)
    d_eff = int(min(d, A.shape[0] - 1))
    emb = SpectralEmbedding(
        n_components=d_eff,
        affinity="precomputed",
        random_state=seed,
    ).fit_transform(A)
    if d_eff < d:  # pad only for tiny datasets where d >= n_cells
        emb = np.pad(emb, ((0, 0), (0, d - d_eff)))
    return emb.astype(np.float32)


def dmkcn_block_b(
    adata,
    gene_list,
    n_clusters: int,
    d: int = 32,
    seed: int = 0,
    full_training: bool = True,
    pretrain_epochs: int = 300,
    n_iter: int = 200,
    lambda1: float = 0.1,  # kernel loss weight   (frozen tuned config)
    lambda2: float = 1.0,  # clustering loss weight
    lambda3: float = 0.05,  # ZINB loss weight
    zinb_on_counts: bool = True,
    nonneg: str = "clip",
    verbose: bool = False,
) -> np.ndarray:
    """scDMKC block-B replacement for one GFM. Returns a (n_cells, d) embedding.

    full_training=True runs the complete scDMKC-style training (pretrain + joint
    self-supervision). full_training=False runs pretraining only (faster; the
    joint phase can be toggled back on later). fit() is called WITHOUT labels, so
    no ground truth ever enters training.
    """
    data = from_scmug_anndata(adata, gene_subset=gene_list)
    X_zinb = data.X_raw if zinb_on_counts else data.X_input
    if not zinb_on_counts and np.any(X_zinb < 0):
        raise ValueError(
            "zinb_on_counts=False would feed negative scaled values to the ZINB "
            "likelihood. Use raw counts for ZINB or disable the ZINB branch in a "
            "dedicated ablation."
        )

    trainer = ScDMKCTrainer(
        n_clusters=n_clusters,
        encoder_hidden=(500, 500, 2000, 10),
        decoder_hidden=(2000, 500, 500),
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        pretrain_epochs=pretrain_epochs,
        n_iter=(n_iter if full_training else 0),
        lr=1e-4,
        pretrain_lr=1e-3,
        min_iter=100,
        tol=1e-4,
        update_interval=3,
        seed=seed,
        verbose=verbose,
    )
    trainer.fit(data.X_input, X_zinb, data.size_factors)  # no y -> no label leakage

    K = trainer.kernel_representation_  # (n_cells, n_cells)
    return spectral_embedding_from_K(K, d=d, seed=seed, nonneg=nonneg)
