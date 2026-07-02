"""Ablation: is scMUG's block C/D useful given the scDMKC block-B representation?

Compares, on the SAME per-GFM embeddings produced by scDMKC block B:

  - COMPLETE             : block B -> block C/D full scMUG pipeline
  - spectral / concat    : concat GFM embeddings -> spectral clustering
  - kmeans / concat      : concat GFM embeddings -> k-means
  - spectral / each GFM  : spectral clustering on each GFM separately
  - spectral / mean GFM  : mean performance across GFMs
  - spectral / oracle GFM: best GFM selected using labels; diagnostic only

No retraining needed: scMUG already dumps ./outputs/<db>_latents.joblib.

Example:
    python ablation_blockB.py \
        --dataset muraro \
        --latents ./outputs/muraro_latents.joblib \
        --complete ./outputs/muraros_scmug_dmkcn.txt
"""

import argparse
import os
import re
from collections import defaultdict

import joblib
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

from utils import calc_acc, calc_ari, calc_nmi, lab2fac, load_data, preprocess


DEFAULT_SEEDS = "1111,2222,3333,4444,5555,6666,7777,8888,9999,10000"


def metrics(y, labels):
    """Return metrics in fixed order: (NMI, ARI, ACC)."""
    return calc_nmi(y, labels), calc_ari(y, labels), calc_acc(y, labels)


def clean_Z(Z):
    """Convert embedding to finite float array."""
    return np.nan_to_num(
        np.asarray(Z, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def spectral(Z, n_clusters, knn, seed):
    """Spectral clustering on a kNN graph built from Z."""
    Z = clean_Z(Z)

    if knn >= Z.shape[0]:
        raise ValueError(f"knn={knn} must be < n_cells={Z.shape[0]}")

    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=knn,
        assign_labels="kmeans",
        random_state=seed,
    ).fit_predict(Z)


def kmeans(Z, n_clusters, seed):
    """K-means directly on Z."""
    Z = clean_Z(Z)

    return KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
    ).fit_predict(Z)


def parse_complete(path):
    """Parse a scMUG-format results file.

    Returns:
        best_by_seed: array of oracle-best rows per seed, selected by max NMI.
        all_rows: all parsed rows.

    Row order is always:
        (NMI, ARI, ACC)
    """
    byseed = defaultdict(list)

    pattern = re.compile(
        r"round:(\d+).*?acc:\s*([\d.]+)\s*\tari:\s*([\d.]+)\s*\tnmi:\s*([\d.]+)"
    )

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            seed, acc, ari, nmi = m.groups()
            byseed[int(seed)].append((float(nmi), float(ari), float(acc)))

    if not byseed:
        raise ValueError(f"No scMUG-format result rows found in: {path}")

    best_by_seed = []
    all_rows = []

    for seed in sorted(byseed):
        rows = byseed[seed]
        all_rows.extend(rows)
        best_by_seed.append(max(rows, key=lambda r: r[0]))  # oracle max NMI

    return np.asarray(best_by_seed), np.asarray(all_rows)


def agg(name, per_seed):
    """Print mean ± std over seeds for rows ordered as (NMI, ARI, ACC)."""
    a = np.asarray(per_seed, dtype=float)

    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name}: expected array with shape (n, 3), got {a.shape}")

    print(
        f"  {name:<30} "
        f"NMI {a[:, 0].mean():.4f}±{a[:, 0].std():.3f}  "
        f"ARI {a[:, 1].mean():.4f}±{a[:, 1].std():.3f}  "
        f"ACC {a[:, 2].mean():.4f}±{a[:, 2].std():.3f}"
    )

    return a


def write_result(fout, seed, method, m):
    """Write one result row. m order: (NMI, ARI, ACC)."""
    fout.write(
        f"seed:{seed}\tmethod:{method}\t"
        f"nmi:{m[0]:.6f}\tari:{m[1]:.6f}\tacc:{m[2]:.6f}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="muraro")
    ap.add_argument("--latents", default="./outputs/muraro_latents.joblib")
    ap.add_argument(
        "--complete",
        default=None,
        help="Optional scMUG-format .txt file of the full pipeline for reference.",
    )
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--outfile", default="./outputs/muraro_ablation_blockB.txt")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    expr_df, cell_type = load_data(args.dataset)
    adata = preprocess(
        expr_df=expr_df.astype(float),
        cell_type=cell_type,
        highly_genes=8000,
    )

    y = lab2fac(adata.obs["cell_type"].to_numpy())
    n_clusters = len(set(y))

    latents = joblib.load(args.latents)

    if len(seeds) != len(latents):
        raise ValueError(
            f"Number of seeds ({len(seeds)}) does not match number of latent "
            f"entries ({len(latents)}). Pass --seeds matching the latent dump."
        )

    first_shape = np.asarray(latents[0]).shape

    print(
        f"{args.dataset}: {n_clusters} types | "
        f"{len(latents)} seed-entries | "
        f"latent shape {first_shape} | "
        f"knn={args.knn}\n"
    )

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

    spec_cat = []
    km_cat = []
    gfm_oracle_best = []
    gfm_mean = []

    with open(args.outfile, "w", encoding="utf-8") as fout:
        fout.write("# metric order: nmi, ari, acc\n")
        fout.write("# spectral_oracle_best_gfm uses labels and is diagnostic only\n")

        for si, latent_val in enumerate(latents):
            seed = seeds[si]
            latent_val = np.asarray(latent_val)

            if latent_val.ndim != 3:
                raise ValueError(
                    f"Expected latent shape (n_cells, n_gfm, d), got {latent_val.shape}"
                )

            n_cells, n_gfm, d = latent_val.shape

            if args.knn >= n_cells:
                raise ValueError(f"knn={args.knn} must be < n_cells={n_cells}")

            if len(y) != n_cells:
                raise ValueError(
                    f"Label length ({len(y)}) does not match latent cells ({n_cells})"
                )

            Z = latent_val.reshape(n_cells, -1)

            m_spec = metrics(y, spectral(Z, n_clusters, args.knn, seed=seed))
            m_km = metrics(y, kmeans(Z, n_clusters, seed=seed))

            spec_cat.append(m_spec)
            km_cat.append(m_km)

            write_result(fout, seed, "spectral_concat", m_spec)
            write_result(fout, seed, "kmeans_concat", m_km)

            per_gfm = []

            for c in range(n_gfm):
                mc = metrics(
                    y,
                    spectral(
                        latent_val[:, c, :],
                        n_clusters,
                        args.knn,
                        seed=seed,
                    ),
                )
                per_gfm.append(mc)
                write_result(fout, seed, f"spectral_gfm{c + 1}", mc)

            per_gfm = np.asarray(per_gfm)

            oracle_best = per_gfm[np.argmax(per_gfm[:, 0])]
            mean_gfm = per_gfm.mean(axis=0)

            gfm_oracle_best.append(oracle_best)
            gfm_mean.append(mean_gfm)

            write_result(fout, seed, "spectral_oracle_best_gfm", oracle_best)
            write_result(fout, seed, "spectral_mean_gfm", mean_gfm)

    print("### Ablation, mean ± std over seeds")
    agg("spectral / concat GFM", spec_cat)
    agg("k-means  / concat GFM", km_cat)
    agg("spectral / mean GFM", gfm_mean)
    agg("spectral / oracle-best GFM", gfm_oracle_best)

    if args.complete:
        best, allrows = parse_complete(args.complete)

        print("\n### COMPLETE pipeline, block C/D reference")
        print(
            f"  oracle-best/seed            "
            f"NMI {best[:, 0].mean():.4f}  "
            f"ARI {best[:, 1].mean():.4f}  "
            f"ACC {best[:, 2].mean():.4f}"
        )
        print(
            f"  all-configs mean            "
            f"NMI {allrows[:, 0].mean():.4f}  "
            f"ARI {allrows[:, 1].mean():.4f}  "
            f"ACC {allrows[:, 2].mean():.4f}"
        )

        print(
            "\nInterpretation:\n"
            "  - If spectral/concat is close to COMPLETE all-configs mean, C/D may add little.\n"
            "  - If COMPLETE all-configs mean is clearly higher, C/D likely helps.\n"
            "  - COMPLETE oracle-best/seed is optimistic because it selects alpha/beta using labels.\n"
            "  - spectral/oracle-best GFM is also optimistic and diagnostic only."
        )

    print(f"\nPer-run results written to {args.outfile}")


if __name__ == "__main__":
    main()
