"""scRNA-seq preprocessing for scDMKC (Section 4.1).

Steps:
  1. compute per-cell size factors from the full raw library;
  2. filter genes expressed in <1% of cells;
  3. library-size normalisation to the median full-library total, then log1p;
  4. selection of the top ``n_top_genes`` highly variable genes;
  5. per-gene z-score scaling for the encoder input.

The ZINB decoder needs raw counts and per-cell size factors, so both are returned
alongside the scaled input.

# ASSUMPTION: the paper writes the normalisation as ln(m(X) * x / sum_o x_io)
# without a "+1"; taken literally this diverges at zeros. We use the standard
# log1p form (as scMUG itself does), i.e. ln(1 + m(X) * x / sum). This is the
# universally used convention.
# ASSUMPTION: HVG selection uses scanpy's dispersion-based method when scanpy is
# available (as stated in the paper); otherwise a numpy dispersion fallback is
# used so the module has no hard scanpy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PreprocessedData:
    X_input: np.ndarray  # (N, t) scaled log-normalised HVG matrix
    X_raw: np.ndarray  # (N, t) raw counts of selected HVGs
    size_factors: np.ndarray  # (N, 1) full-library per-cell size factors
    gene_index: np.ndarray  # selected HVG indices in original gene axis


def _select_hvg_numpy(log_norm: np.ndarray, n_top: int) -> np.ndarray:
    """Dispersion-based HVG selection fallback."""
    mean = log_norm.mean(axis=0)
    var = log_norm.var(axis=0)

    dispersion = np.zeros_like(mean, dtype=np.float32)
    nz = mean > 0
    dispersion[nz] = var[nz] / mean[nz]

    order = np.argsort(dispersion)[::-1]
    return np.sort(order[:n_top])


def _compute_size_factors_from_full_counts(
    counts_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute full-library cell totals and size factors."""
    cell_totals_full = counts_full.sum(axis=1, keepdims=True).astype(np.float32)
    cell_totals_full[cell_totals_full == 0] = 1.0

    median_total = float(np.median(cell_totals_full))
    if median_total <= 0:
        median_total = 1.0

    size_factors = (cell_totals_full / median_total).astype(np.float32)
    return cell_totals_full, size_factors, median_total


def preprocess(
    counts: np.ndarray,
    n_top_genes: int = 1000,
    min_cell_fraction: float = 0.01,
    use_scanpy: bool = True,
    round_counts: bool = True,
) -> PreprocessedData:
    """Run the scDMKC preprocessing pipeline on a cells x genes count matrix."""
    counts_full = np.asarray(counts, dtype=np.float32)
    if counts_full.ndim != 2:
        raise ValueError("counts must be a 2D array with shape (cells, genes).")

    n_cells = counts_full.shape[0]

    # 1. size factors from the full raw library -----------------------------
    cell_totals_full, size_factors, median_total = (
        _compute_size_factors_from_full_counts(counts_full)
    )

    # 2. gene filtering -----------------------------------------------------
    nonzero_cells_per_gene = (counts_full > 0).sum(axis=0)
    keep = nonzero_cells_per_gene > (min_cell_fraction * n_cells)

    if not np.any(keep):
        raise ValueError("No genes remain after filtering.")

    counts_filt = counts_full[:, keep]
    kept_idx = np.where(keep)[0]

    # 3. normalisation using full-library totals ----------------------------
    log_norm = np.log1p(median_total * counts_filt / cell_totals_full).astype(
        np.float32
    )

    # 4. HVG selection ------------------------------------------------------
    n_top = min(n_top_genes, log_norm.shape[1])

    hvg_local = None
    if use_scanpy:
        try:
            import scanpy as sc
            from anndata import AnnData
        except ImportError:
            hvg_local = None
        else:
            adata = AnnData(log_norm.copy())
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
            hvg_local = np.where(adata.var["highly_variable"].values)[0]

    if hvg_local is None or len(hvg_local) == 0:
        hvg_local = _select_hvg_numpy(log_norm, n_top)

    hvg_local = np.asarray(hvg_local, dtype=int)

    log_norm_hvg = log_norm[:, hvg_local]
    raw_hvg = counts_filt[:, hvg_local]

    if round_counts:
        raw_hvg = np.rint(raw_hvg)

    gene_index = kept_idx[hvg_local]

    # 5. z-score scaling for encoder input ---------------------------------
    mu = log_norm_hvg.mean(axis=0, keepdims=True)
    sd = log_norm_hvg.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0

    x_input = ((log_norm_hvg - mu) / sd).astype(np.float32)

    return PreprocessedData(
        X_input=np.ascontiguousarray(x_input, dtype=np.float32),
        X_raw=np.ascontiguousarray(raw_hvg, dtype=np.float32),
        size_factors=np.ascontiguousarray(size_factors, dtype=np.float32),
        gene_index=np.asarray(gene_index, dtype=int),
    )


def from_scmug_anndata(adata, gene_subset=None) -> PreprocessedData:
    """Build PreprocessedData from a scMUG / scDeepCluster-style AnnData.

    Expects:
        adata.X      -> scaled log-normalised HVG matrix
        adata.raw.X  -> raw counts, possibly with a different gene axis

    If adata.raw is present, raw counts are aligned by gene names.
    """
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    var_names = np.asarray(adata.var_names, dtype=object)

    if adata.raw is not None:
        raw_all = adata.raw.X
        raw_all = (
            raw_all.toarray() if hasattr(raw_all, "toarray") else np.asarray(raw_all)
        )
        raw_var_names = np.asarray(adata.raw.var_names, dtype=object)

        raw_pos = {str(g): i for i, g in enumerate(raw_var_names)}
        raw_idx = np.array(
            [raw_pos[str(g)] for g in var_names if str(g) in raw_pos],
            dtype=int,
        )
        x_idx = np.array(
            [i for i, g in enumerate(var_names) if str(g) in raw_pos],
            dtype=int,
        )

        if raw_idx.size != len(var_names):
            missing = [str(g) for g in var_names if str(g) not in raw_pos]
            raise ValueError(
                f"adata.raw is missing {len(missing)} genes from adata.var_names. "
                f"First missing genes: {missing[:10]}"
            )

        raw = raw_all[:, raw_idx]
        X = X[:, x_idx]
        var_names = var_names[x_idx]

        raw_full_for_sf = raw_all

    elif "counts" in getattr(adata, "layers", {}):
        raw = adata.layers["counts"]
        raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)

        if raw.shape[1] != len(var_names):
            raise ValueError(
                "adata.layers['counts'] must have the same gene axis as adata.var_names."
            )

        raw_full_for_sf = raw

    else:
        raise ValueError("No raw counts found in adata.raw or adata.layers['counts'].")

    # size factors from full raw library when possible ----------------------
    if "size_factors" in getattr(adata, "obs", {}):
        sf = np.asarray(adata.obs["size_factors"], dtype=np.float32).reshape(-1, 1)
    else:
        _, sf, _ = _compute_size_factors_from_full_counts(
            np.asarray(raw_full_for_sf, dtype=np.float32)
        )

    # optional gene subsetting ----------------------------------------------
    if gene_subset is not None:
        wanted = set(map(str, gene_subset))
        idx = np.array(
            [i for i, g in enumerate(var_names) if str(g) in wanted], dtype=int
        )

        if idx.size == 0:
            raise ValueError("None of gene_subset was found in adata.var_names.")

        X = X[:, idx]
        raw = raw[:, idx]
        var_names = var_names[idx]
    else:
        idx = np.arange(len(var_names), dtype=int)

    return PreprocessedData(
        X_input=np.ascontiguousarray(X, dtype=np.float32),
        X_raw=np.ascontiguousarray(raw, dtype=np.float32),
        size_factors=np.ascontiguousarray(sf, dtype=np.float32),
        gene_index=idx,
    )
