"""
Ablation indépendante des blocs C et D de scMUG.

Compare deux représentations de bloc B:
  1. scMUG standard
  2. scMUG + DMKCN injecté dans le bloc B

Le script ne réentraîne rien.
Il réutilise les .joblib contenant les latents de sortie du bloc B.

Formats acceptés pour les .joblib:
  - ndarray: (n_cells, n_gfm, latent_dim)
  - ndarray: (n_cells, latent_dim)
  - dict contenant une clé parmi:
      "Z", "z", "latents", "latent", "embeddings", "embedding", "features", "X"

Sorties:
  - TSV détaillé
  - TSV résumé
  - TSV meilleurs scores par modèle
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

from sklearn.cluster import (
    KMeans,
    SpectralClustering,
    AgglomerativeClustering,
)
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from utils import load_data, preprocess, lab2fac


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def clustering_acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)

    true_map = {v: i for i, v in enumerate(true_classes)}
    pred_map = {v: i for i, v in enumerate(pred_classes)}

    yt = np.array([true_map[v] for v in y_true])
    yp = np.array([pred_map[v] for v in y_pred])

    n = max(yt.max(), yp.max()) + 1
    cost = np.zeros((n, n), dtype=np.int64)

    for i in range(len(yt)):
        cost[yp[i], yt[i]] += 1

    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    return cost[row_ind, col_ind].sum() / len(yt)


def compute_metrics(y_true, y_pred):
    return {
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "ACC": clustering_acc(y_true, y_pred),
    }


# ---------------------------------------------------------------------
# Latent loading
# ---------------------------------------------------------------------


def extract_latents(obj):
    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, dict):
        keys = [
            "Z",
            "z",
            "latents",
            "latent",
            "embeddings",
            "embedding",
            "features",
            "X",
        ]

        for k in keys:
            if k in obj:
                return np.asarray(obj[k])

        raise ValueError(
            "Impossible de trouver les latents dans le dict joblib. "
            f"Clés disponibles: {list(obj.keys())}"
        )

    if isinstance(obj, (list, tuple)):
        for item in obj:
            try:
                return extract_latents(item)
            except Exception:
                pass

    raise ValueError(f"Format joblib non supporté: {type(obj)}")


def load_latents(path):
    obj = joblib.load(path)
    Z = np.asarray(extract_latents(obj))

    if Z.ndim == 2:
        # n_cells × latent_dim -> n_cells × 1 × latent_dim
        Z = Z[:, None, :]

    if Z.ndim != 3:
        raise ValueError(f"Latents attendus en 2D ou 3D, mais shape obtenue: {Z.shape}")

    if not np.all(np.isfinite(Z)):
        raise ValueError(f"NaN ou Inf détectés dans {path}")

    return Z.astype(np.float32)


def flatten_latents(Z):
    return Z.reshape(Z.shape[0], -1)


# ---------------------------------------------------------------------
# Bloc C-like matrices
# ---------------------------------------------------------------------


def build_global_similarity(Z):
    """
    Similarité globale sur les latents fusionnés.
    Approximation raisonnable du signal global de scMUG.
    """
    X = flatten_latents(Z)
    X = StandardScaler().fit_transform(X)

    S = cosine_similarity(X)
    S = (S + 1.0) / 2.0
    S = np.maximum(S, 0.0)
    np.fill_diagonal(S, 1.0)

    return S.astype(np.float32)


def build_local_coclustering(Z, n_clusters, kmeans_times, seed):
    """
    Similarité locale:
      - KMeans répété par GFM/canal
      - deux cellules sont proches si elles co-clusterisent souvent.
    """
    n_cells, n_gfm, _ = Z.shape
    S = np.zeros((n_cells, n_cells), dtype=np.float32)

    rng = np.random.default_rng(seed)
    km_seeds = rng.integers(0, 10_000_000, size=(n_gfm, kmeans_times))

    for g in range(n_gfm):
        Xg = StandardScaler().fit_transform(Z[:, g, :])

        for t in range(kmeans_times):
            labels = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=int(km_seeds[g, t]),
            ).fit_predict(Xg)

            for k in range(n_clusters):
                idx = np.where(labels == k)[0]
                if len(idx) > 0:
                    S[np.ix_(idx, idx)] += 1.0

    S /= float(n_gfm * kmeans_times)
    np.fill_diagonal(S, 1.0)

    return S.astype(np.float32)


def combine_C(S_global, S_local, alpha, beta):
    S = alpha * S_global + beta * S_local
    S = np.maximum(S, 0.0)
    np.fill_diagonal(S, 1.0)

    max_val = S.max()
    if max_val > 0:
        S = S / max_val

    return S.astype(np.float32)


# ---------------------------------------------------------------------
# Clustering variants
# ---------------------------------------------------------------------


def cluster_B_kmeans(Z, n_clusters, seed):
    """
    Sans C, sans D:
      B -> concat -> KMeans
    """
    X = flatten_latents(Z)
    X = StandardScaler().fit_transform(X)

    return KMeans(
        n_clusters=n_clusters,
        n_init=50,
        random_state=seed,
    ).fit_predict(X)


def cluster_B_knn_spectral(Z, n_clusters, seed, n_neighbors):
    """
    Sans C scMUG, avec D-like spectral:
      B -> concat -> kNN graph -> Spectral
    """
    X = flatten_latents(Z)
    X = StandardScaler().fit_transform(X)

    n_neighbors = min(n_neighbors, X.shape[0] - 1)

    A = kneighbors_graph(
        X,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=True,
    )

    A = 0.5 * (A + A.T)
    A = A.toarray().astype(np.float32)
    np.fill_diagonal(A, 1.0)

    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    ).fit_predict(A)


def cluster_C_spectral(S, n_clusters, seed):
    """
    Avec C, avec D:
      S_C -> SpectralClustering
    """
    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    ).fit_predict(S)


def cluster_C_kmeans_rows(S, n_clusters, seed):
    """
    Avec C, sans D:
      chaque ligne de S est utilisée comme embedding d'affinité,
      puis KMeans.
    """
    X = StandardScaler().fit_transform(S)

    return KMeans(
        n_clusters=n_clusters,
        n_init=50,
        random_state=seed,
    ).fit_predict(X)


def cluster_C_agglomerative_distance(S, n_clusters):
    """
    Avec C, sans D:
      distance = 1 - affinité,
      puis clustering agglomératif.
    """
    D = 1.0 - S
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        )

    return model.fit_predict(D)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def add_result(
    rows, model_name, condition, block_C, block_D, seed, alpha, beta, y, pred
):
    m = compute_metrics(y, pred)

    rows.append(
        {
            "model": model_name,
            "condition": condition,
            "block_C": block_C,
            "block_D": block_D,
            "seed": seed,
            "alpha": alpha,
            "beta": beta,
            "NMI": m["NMI"],
            "ARI": m["ARI"],
            "ACC": m["ACC"],
        }
    )


def evaluate_model(
    model_name,
    latent_path,
    y,
    n_clusters,
    seeds,
    alphas,
    betas,
    kmeans_times,
    n_neighbors,
):
    print(f"\n=== {model_name} ===")
    print(f"Loading: {latent_path}")

    Z = load_latents(latent_path)

    print(f"Latent shape: {Z.shape}")

    if Z.shape[0] != len(y):
        raise ValueError(
            f"{model_name}: mismatch cellules. Z.shape[0]={Z.shape[0]}, len(y)={len(y)}"
        )

    rows = []

    for seed in seeds:
        print(f"Seed {seed}")

        # 1. Pas C, pas D
        pred = cluster_B_kmeans(Z, n_clusters, seed)
        add_result(
            rows,
            model_name,
            "B_only__no_C__no_D__kmeans",
            "absent",
            "absent",
            seed,
            np.nan,
            np.nan,
            y,
            pred,
        )

        # 2. Pas C scMUG, D spectral sur graphe kNN simple
        pred = cluster_B_knn_spectral(Z, n_clusters, seed, n_neighbors)
        add_result(
            rows,
            model_name,
            "B_only__no_C__D_spectral_knn",
            "absent",
            "present",
            seed,
            np.nan,
            np.nan,
            y,
            pred,
        )

        # Pré-calcul C pour la seed
        S_global = build_global_similarity(Z)
        S_local = build_local_coclustering(
            Z,
            n_clusters=n_clusters,
            kmeans_times=kmeans_times,
            seed=seed,
        )

        # 3. Global only + D
        pred = cluster_C_spectral(S_global, n_clusters, seed)
        add_result(
            rows,
            model_name,
            "B_plus_C_global_only__D_spectral",
            "global_only",
            "present",
            seed,
            1.0,
            0.0,
            y,
            pred,
        )

        # 4. Local only + D
        pred = cluster_C_spectral(S_local, n_clusters, seed)
        add_result(
            rows,
            model_name,
            "B_plus_C_local_only__D_spectral",
            "local_only",
            "present",
            seed,
            0.0,
            1.0,
            y,
            pred,
        )

        # 5. C complet avec sweep alpha/beta
        for alpha in alphas:
            for beta in betas:
                if alpha == 0 and beta == 0:
                    continue

                S = combine_C(S_global, S_local, alpha, beta)

                # C + D
                pred = cluster_C_spectral(S, n_clusters, seed)
                add_result(
                    rows,
                    model_name,
                    "B_plus_C__D_spectral",
                    "present",
                    "present",
                    seed,
                    alpha,
                    beta,
                    y,
                    pred,
                )

                # C sans D: KMeans sur lignes de S
                pred = cluster_C_kmeans_rows(S, n_clusters, seed)
                add_result(
                    rows,
                    model_name,
                    "B_plus_C__no_D__kmeans_on_affinity",
                    "present",
                    "absent",
                    seed,
                    alpha,
                    beta,
                    y,
                    pred,
                )

                # C sans D: agglomératif sur distance 1-S
                pred = cluster_C_agglomerative_distance(S, n_clusters)
                add_result(
                    rows,
                    model_name,
                    "B_plus_C__no_D__agglomerative_on_distance",
                    "present",
                    "absent",
                    seed,
                    alpha,
                    beta,
                    y,
                    pred,
                )

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="muraro", type=str)

    parser.add_argument(
        "--latents-scmug",
        default="outputs/muraro_latents_scmug.joblib",
        type=str,
    )
    parser.add_argument(
        "--latents-dmkcn",
        default="outputs/muraro_latents_scmug_dmkcn.joblib",
        type=str,
    )

    parser.add_argument(
        "--outfile",
        default="outputs/ablation_c_d_compare_muraro.tsv",
        type=str,
    )

    parser.add_argument("--cluster-number", default=None, type=int)
    parser.add_argument(
        "--seeds",
        default="1111,2222,3333,4444,5555",
        type=str,
    )
    parser.add_argument("--kmeans-times", default=20, type=int)
    parser.add_argument("--n-neighbors", default=10, type=int)

    parser.add_argument(
        "--alphas",
        default="0.0,0.25,0.5,0.75,1.0",
        type=str,
    )
    parser.add_argument(
        "--betas",
        default="0.0,0.25,0.5,0.75,1.0",
        type=str,
    )

    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    betas = [float(x) for x in args.betas.split(",")]

    print(f"Dataset: {args.dataset}")

    expr_df, cell_type = load_data(args.dataset)
    expr_df = expr_df.astype(float)

    adata = preprocess(
        expr_df=expr_df,
        cell_type=cell_type,
        highly_genes=8000,
    )

    y = lab2fac(adata.obs["cell_type"].to_numpy())

    if args.cluster_number is None:
        n_clusters = len(np.unique(y))
    else:
        n_clusters = args.cluster_number

    print(f"n_cells: {len(y)}")
    print(f"n_clusters: {n_clusters}")

    all_rows = []

    all_rows.extend(
        evaluate_model(
            model_name="scMUG",
            latent_path=args.latents_scmug,
            y=y,
            n_clusters=n_clusters,
            seeds=seeds,
            alphas=alphas,
            betas=betas,
            kmeans_times=args.kmeans_times,
            n_neighbors=args.n_neighbors,
        )
    )

    all_rows.extend(
        evaluate_model(
            model_name="scMUG_DMKCN",
            latent_path=args.latents_dmkcn,
            y=y,
            n_clusters=n_clusters,
            seeds=seeds,
            alphas=alphas,
            betas=betas,
            kmeans_times=args.kmeans_times,
            n_neighbors=args.n_neighbors,
        )
    )

    df = pd.DataFrame(all_rows)

    outdir = os.path.dirname(args.outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    df.to_csv(args.outfile, sep="\t", index=False)

    summary = (
        df.groupby(["model", "condition"])[["NMI", "ARI", "ACC"]]
        .agg(["mean", "std", "max"])
        .reset_index()
    )

    summary_outfile = args.outfile.replace(".tsv", "_summary.tsv")
    summary.to_csv(summary_outfile, sep="\t", index=False)

    best = (
        df.sort_values("ARI", ascending=False)
        .groupby(["model", "condition"], as_index=False)
        .head(1)
        .sort_values(["model", "ARI"], ascending=[True, False])
    )

    best_outfile = args.outfile.replace(".tsv", "_best.tsv")
    best.to_csv(best_outfile, sep="\t", index=False)

    print("\nSaved files:")
    print(args.outfile)
    print(summary_outfile)
    print(best_outfile)

    print("\nBest ARI per model/condition:")
    print(
        best[
            [
                "model",
                "condition",
                "ARI",
                "NMI",
                "ACC",
                "seed",
                "alpha",
                "beta",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
