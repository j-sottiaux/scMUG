"""Integration adapter: use scDMKC as scMUG's per-GFM representation generator.

Drop-in replacement for scMUG's ZINB-autoencoder block B. For one gene functional
module (GFM), it trains scDMKC on that module's genes, takes the consistent kernel
representation K, and returns a spectral embedding of K that scMUG's block C
consumes exactly like the original 32-D autoencoder latent.

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

2. ZINB target confound: scMUG's original block B feeds the *scaled* adata.X as the
   ZINB target; this adapter feeds the true integer counts (adata.raw.X) as scDMKC
   intends. If the integration improves results, part of the gain may come from
   this corrected target rather than the kernels alone. Set zinb_on_counts=False to
   mirror scMUG's original (scaled-target) behaviour for a cleaner ablation.
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
    S = (K + K.T) / 2.0
    if nonneg == "clip":
        A = np.clip(S, 0.0, None)
    elif nonneg == "abs":
        A = np.abs(S)
    elif nonneg == "shift":
        A = S - S.min()
    else:
        raise ValueError(f"unknown nonneg='{nonneg}'")
    return A.astype(np.float64)


def spectral_embedding_from_K(
    K: np.ndarray, d: int = 32, seed: int = 0, nonneg: str = "clip"
) -> np.ndarray:
    """Return a (n_cells, d) spectral embedding of the kernel representation K.

    Fails fast on malformed input rather than letting a corrupt embedding
    silently pollute scMUG's block C.
    """
    # (1) input validation
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")
    if K.shape[0] <= 2:
        raise ValueError("Need at least 3 cells for spectral embedding.")

    A = affinity_from_K(K, nonneg=nonneg)

    # (2) a diverged dmkcn run can leave NaN/Inf in K -> catch it loudly here
    if not np.isfinite(A).all():
        raise ValueError("Affinity matrix contains NaN or Inf.")
    # (3) degenerate all-zero affinity (e.g. nonneg='clip' with K <= 0 everywhere)
    if np.all(A == 0):
        raise ValueError("Affinity matrix is all zeros after transformation.")

    d_eff = int(min(d, A.shape[0] - 1))
    emb = SpectralEmbedding(
        n_components=d_eff, affinity="precomputed", random_state=seed,
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
    n_iter: int = 300,
    lambda1: float = 0.1,   # kernel loss weight   (frozen tuned config)
    lambda2: float = 1.0,   # clustering loss weight
    lambda3: float = 0.05,  # ZINB loss weight
    zinb_on_counts: bool = True,
    nonneg: str = "clip",
    verbose: bool = False,
) -> np.ndarray:
    """scDMKC block-B replacement for one GFM. Returns a (n_cells, d) embedding.

    full_training=True runs the complete scDMKC (pretrain + joint self-supervision),
    faithful to the paper. full_training=False runs pretraining only (faster; the
    joint phase can be toggled back on later). fit() is called WITHOUT labels, so
    no ground truth ever enters training.
    """
    data = from_scmug_anndata(adata, gene_subset=gene_list)
    X_zinb = data.X_raw if zinb_on_counts else data.X_input

    trainer = ScDMKCTrainer(
        n_clusters=n_clusters,
        encoder_hidden=(500, 500, 2000, 10),
        decoder_hidden=(2000, 500, 500),
        lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
        pretrain_epochs=pretrain_epochs,
        n_iter=(n_iter if full_training else 0),
        lr=1e-4, pretrain_lr=1e-3,
        min_iter=100, tol=1e-4, update_interval=3,
        seed=seed, verbose=verbose,
    )
    trainer.fit(data.X_input, X_zinb, data.size_factors)  # no y -> no label leakage

    K = trainer.kernel_representation_        # (n_cells, n_cells)
    return spectral_embedding_from_K(K, d=d, seed=seed, nonneg=nonneg)
