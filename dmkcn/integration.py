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
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import normalize

from computation_metrics import synchronized_elapsed, synchronized_start
from .preprocessing import from_scmug_anndata
from .trainer import ScDMKCTrainer


SUPPORTED_PROJECTIONS = (
    "spectral_dense",
    "spectral_knn",
    "svd_raw",
    "svd_l2",
    "svd_zscore",
)


def embedding_key(projection: str, d: int) -> str:
    """Stable key used in joblib filenames and diagnostics."""
    return f"{projection}_d{int(d)}"


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
    return A.astype(np.float32, copy=False)


def kernel_diagnostics(K: np.ndarray) -> dict[str, float]:
    """Compact diagnostics for the learned cell-cell representation."""
    K = np.asarray(K, dtype=np.float32)
    row_l1 = np.abs(K).sum(axis=1)
    return {
        "min": float(K.min()),
        "max": float(K.max()),
        "mean": float(K.mean()),
        "std": float(K.std()),
        "positive_fraction": float(np.mean(K > 0)),
        "negative_fraction": float(np.mean(K < 0)),
        "asymmetry_max": float(np.max(np.abs(K - K.T))),
        "row_l1_min": float(row_l1.min()),
        "row_l1_median": float(np.median(row_l1)),
        "row_l1_max": float(row_l1.max()),
    }


def knn_affinity_from_K(
    K: np.ndarray,
    n_neighbors: int,
    nonneg: str = "clip",
    row_chunk_size: int = 256,
) -> tuple[sparse.csr_matrix, dict[str, float | int]]:
    """Build a weighted symmetric kNN graph from the affinity induced by K."""
    A = affinity_from_K(K, nonneg=nonneg)
    n_cells = A.shape[0]
    if not 1 <= n_neighbors < n_cells:
        raise ValueError(
            f"n_neighbors must be in [1, {n_cells - 1}], got {n_neighbors}."
        )
    if row_chunk_size <= 0:
        raise ValueError(f"row_chunk_size must be positive, got {row_chunk_size}.")

    np.fill_diagonal(A, 0.0)
    row_parts = []
    col_parts = []
    data_parts = []
    kth = n_cells - n_neighbors
    for start in range(0, n_cells, row_chunk_size):
        stop = min(start + row_chunk_size, n_cells)
        block = A[start:stop]
        indices = np.argpartition(block, kth=kth, axis=1)[:, -n_neighbors:]
        values = np.take_along_axis(block, indices, axis=1)
        row_parts.append(np.repeat(np.arange(start, stop), n_neighbors))
        col_parts.append(indices.reshape(-1))
        data_parts.append(values.reshape(-1))

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.concatenate(data_parts)
    keep = data > 0
    graph = sparse.csr_matrix(
        (data[keep], (rows[keep], cols[keep])),
        shape=(n_cells, n_cells),
        dtype=np.float32,
    )
    graph = graph.maximum(graph.T).tocsr()
    graph.eliminate_zeros()
    n_components, _ = connected_components(graph, directed=False)
    degree = np.asarray((graph > 0).sum(axis=1)).ravel()
    diagnostics = {
        "n_neighbors_requested": int(n_neighbors),
        "topk_row_chunk_size": int(row_chunk_size),
        "n_edges_undirected": int(graph.nnz // 2),
        "n_connected_components": int(n_components),
        "degree_min": int(degree.min()),
        "degree_median": float(np.median(degree)),
        "degree_max": int(degree.max()),
    }
    return graph, diagnostics


def spectral_embedding_from_K(
    K: np.ndarray,
    d: int = 32,
    seed: int = 0,
    nonneg: str = "clip",
    graph_neighbors: int | None = None,
) -> np.ndarray:
    """Return a (n_cells, d) spectral embedding of the kernel representation K."""
    if d <= 0:
        raise ValueError(f"d must be strictly positive, got {d}.")
    if graph_neighbors is None:
        A = affinity_from_K(K, nonneg=nonneg)
    else:
        A, _ = knn_affinity_from_K(
            K,
            n_neighbors=graph_neighbors,
            nonneg=nonneg,
        )
    d_eff = int(min(d, A.shape[0] - 1))
    emb = SpectralEmbedding(
        n_components=d_eff,
        affinity="precomputed",
        random_state=seed,
    ).fit_transform(A)
    if d_eff < d:  # pad only for tiny datasets where d >= n_cells
        emb = np.pad(emb, ((0, 0), (0, d - d_eff)))
    return emb.astype(np.float32)


def svd_embedding_from_K(
    K: np.ndarray,
    d: int = 32,
    seed: int = 0,
    row_mode: str = "raw",
) -> np.ndarray:
    """Project rows of K while preserving their signed, asymmetric geometry."""
    if d <= 0:
        raise ValueError(f"d must be strictly positive, got {d}.")
    rows = np.asarray(K, dtype=np.float32)
    if row_mode == "raw":
        transformed = rows
    elif row_mode == "l2":
        transformed = normalize(rows, norm="l2", copy=True)
    elif row_mode == "zscore":
        row_mean = rows.mean(axis=1, keepdims=True)
        row_std = rows.std(axis=1, keepdims=True)
        transformed = (rows - row_mean) / np.where(row_std > 0, row_std, 1.0)
    else:
        raise ValueError(f"Unknown SVD row_mode='{row_mode}'.")

    d_eff = int(min(d, min(transformed.shape) - 1))
    embedding = TruncatedSVD(
        n_components=d_eff,
        algorithm="randomized",
        n_iter=7,
        random_state=seed,
    ).fit_transform(transformed)
    if d_eff < d:
        embedding = np.pad(embedding, ((0, 0), (0, d - d_eff)))
    return np.asarray(embedding, dtype=np.float32)


def embeddings_from_K(
    K: np.ndarray,
    projections: tuple[str, ...] | list[str],
    dimensions: tuple[int, ...] | list[int],
    seed: int,
    nonneg: str = "clip",
    graph_neighbors: int = 30,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Create multiple block-B candidates from one trained K."""
    projection_names = tuple(dict.fromkeys(projections))
    dims = tuple(sorted(set(int(value) for value in dimensions)))
    if not projection_names:
        raise ValueError("At least one K projection must be requested.")
    unknown = sorted(set(projection_names) - set(SUPPORTED_PROJECTIONS))
    if unknown:
        raise ValueError(
            f"Unknown projections {unknown}; choose from {SUPPORTED_PROJECTIONS}."
        )
    if not dims or any(value <= 0 for value in dims):
        raise ValueError(f"Embedding dimensions must be positive, got {dims}.")

    max_d = max(dims)
    embeddings: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {}

    for projection in projection_names:
        if projection == "spectral_dense":
            full = spectral_embedding_from_K(
                K,
                d=max_d,
                seed=seed,
                nonneg=nonneg,
            )
            diagnostics[projection] = {"graph": "dense"}
        elif projection == "spectral_knn":
            graph, graph_diagnostics = knn_affinity_from_K(
                K,
                n_neighbors=graph_neighbors,
                nonneg=nonneg,
            )
            d_eff = min(max_d, graph.shape[0] - 1)
            full = SpectralEmbedding(
                n_components=d_eff,
                affinity="precomputed",
                random_state=seed,
            ).fit_transform(graph)
            if d_eff < max_d:
                full = np.pad(full, ((0, 0), (0, max_d - d_eff)))
            full = np.asarray(full, dtype=np.float32)
            diagnostics[projection] = graph_diagnostics
        else:
            row_mode = projection.removeprefix("svd_")
            full = svd_embedding_from_K(K, d=max_d, seed=seed, row_mode=row_mode)
            diagnostics[projection] = {"row_mode": row_mode}

        for d in dims:
            embeddings[embedding_key(projection, d)] = full[:, :d].copy()

    return embeddings, diagnostics


def dmkcn_block_b(
    adata,
    gene_list,
    n_clusters: int,
    d: int = 32,
    seed: int = 0,
    full_training: bool = True,
    pretrain_epochs: int = 300,
    n_iter: int = 200,
    lambda1: float | None = None,
    lambda2: float | None = None,
    lambda3: float | None = None,
    zinb_on_counts: bool = True,
    nonneg: str = "clip",
    allow_pseudo_counts: bool = False,
    projections: tuple[str, ...] | list[str] | None = None,
    embedding_dims: tuple[int, ...] | list[int] | None = None,
    primary_projection: str = "spectral_dense",
    graph_neighbors: int = 30,
    return_artifacts: bool = False,
    verbose: bool = False,
) -> np.ndarray | dict:
    """scDMKC block-B replacement for one GFM. Returns a (n_cells, d) embedding.

    full_training=True runs the complete scDMKC-style training (pretrain + joint
    self-supervision). full_training=False runs pretraining only (faster; the
    joint phase can be toggled back on later). fit() is called WITHOUT labels, so
    no ground truth ever enters training.
    """
    preprocessing_started = synchronized_start()
    data = from_scmug_anndata(
        adata,
        gene_subset=gene_list,
        allow_pseudo_counts=allow_pseudo_counts,
    )
    X_zinb = data.X_raw if zinb_on_counts else data.X_input
    if not zinb_on_counts and np.any(X_zinb < 0):
        raise ValueError(
            "zinb_on_counts=False would feed negative scaled values to the ZINB "
            "likelihood. Use raw counts for ZINB or disable the ZINB branch in a "
            "dedicated ablation."
        )
    preprocessing_seconds = synchronized_elapsed(preprocessing_started)

    training_started = synchronized_start()
    lambda_values = (lambda1, lambda2, lambda3)
    if any(value is None for value in lambda_values) and not all(
        value is None for value in lambda_values
    ):
        raise ValueError(
            "lambda1, lambda2 and lambda3 must either all be provided or all be omitted."
        )
    trainer_kwargs = {}
    if all(value is not None for value in lambda_values):
        trainer_kwargs.update(
            lambda1=float(lambda1),
            lambda2=float(lambda2),
            lambda3=float(lambda3),
        )

    trainer = ScDMKCTrainer(
        n_clusters=n_clusters,
        encoder_hidden=(500, 500, 2000, 10),
        decoder_hidden=(2000, 500, 500),
        pretrain_epochs=pretrain_epochs,
        n_iter=(n_iter if full_training else 0),
        lr=1e-4,
        pretrain_lr=1e-3,
        min_iter=100,
        tol=1e-4,
        update_interval=3,
        seed=seed,
        verbose=verbose,
        **trainer_kwargs,
    )
    trainer.fit(data.X_input, X_zinb, data.size_factors)  # no y -> no label leakage
    training_seconds = synchronized_elapsed(training_started)

    K = trainer.kernel_representation_  # (n_cells, n_cells)
    requested_projections = list(projections or [primary_projection])
    if primary_projection not in requested_projections:
        requested_projections.append(primary_projection)
    requested_dims = list(embedding_dims or [d])
    if d not in requested_dims:
        requested_dims.append(d)

    projection_started = synchronized_start()
    embeddings, projection_diagnostics = embeddings_from_K(
        K,
        projections=requested_projections,
        dimensions=requested_dims,
        seed=seed,
        nonneg=nonneg,
        graph_neighbors=graph_neighbors,
    )
    projection_seconds = synchronized_elapsed(projection_started)
    primary_key = embedding_key(primary_projection, d)
    count_contract = dict(getattr(adata, "uns", {}).get("count_contract", {}))
    artifacts = {
        "timings": {
            "dmkcn_preprocessing": preprocessing_seconds,
            "dmkcn_training": training_seconds,
            "k_projection": projection_seconds,
        },
        "embeddings": embeddings,
        "primary_key": primary_key,
        "labels_K_raw": trainer.labels_.copy(),
        "diagnostics": {
            "input": {
                "n_cells": int(data.X_input.shape[0]),
                "n_genes": int(data.X_input.shape[1]),
                "zinb_on_counts": bool(zinb_on_counts),
                "allow_pseudo_counts": bool(allow_pseudo_counts),
                "source_counts_are_integer": count_contract.get(
                    "counts_are_integer"
                ),
                "size_factor_source": count_contract.get("size_factor_source"),
                "size_factor_min": float(data.size_factors.min()),
                "size_factor_median": float(np.median(data.size_factors)),
                "size_factor_max": float(data.size_factors.max()),
            },
            "kernel": kernel_diagnostics(K),
            "projections": projection_diagnostics,
            "training": trainer.fit_diagnostics_,
        },
    }
    if return_artifacts:
        return artifacts
    return embeddings[primary_key]
