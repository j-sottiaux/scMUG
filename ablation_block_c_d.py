"""Ablation of scMUG blocks C/D from saved block-B latents.

This script avoids proxy similarities. For every saved latent tensor it rebuilds
the same matrices used by scMUG.py:

  - s_g / mat1: repeated K-means co-clustering via accelerate.get_mat1
  - s_d / mat2: local density similarity via accelerate.get_mat2
  - block D: SpectralClustering(affinity="precomputed") on alpha*s_g + beta*s_d

Inputs are the joblib files written by scMUG.py with:
  --block-b autoencoder
  --block-b dmkcn

Each joblib is expected to contain a list with one array per seed, where each
array has shape:

  (n_cells, n_gfm, latent_dim)
"""

import argparse
import os
from collections import defaultdict

import joblib
import numpy as np
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans, SpectralClustering

from accelerate import get_mat1, get_mat2
from utils import (
    calc_acc,
    calc_ari,
    calc_nmi,
    c_kmeans,
    lab2fac,
    load_data,
    preprocess,
    reducer,
    set_seed,
)


DEFAULT_SEEDS = "1111,2222,3333,4444,5555,6666,7777,8888,9999,10000"

DEFAULT_ALPHA_BETA = [
    (0, 1),
    (0.001, 1),
    (0.01, 1),
    (0.1, 1),
    (1, 1),
    (1, 0.1),
    (1, 0.01),
    (1, 0.001),
    (1, 0),
]


def clean_array(x):
    return np.nan_to_num(
        np.asarray(x, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def metrics(y_true, y_pred):
    return (
        calc_nmi(y_true, y_pred),
        calc_ari(y_true, y_pred),
        calc_acc(y_true, y_pred),
    )


def load_latent_list(path, seeds):
    obj = joblib.load(path)

    if isinstance(obj, dict):
        for key in ("latents", "latent", "Z", "z", "embeddings", "features"):
            if key in obj:
                obj = obj[key]
                break

    if isinstance(obj, np.ndarray):
        obj = [obj]

    if not isinstance(obj, (list, tuple)):
        raise TypeError(
            f"{path}: expected list/tuple/ndarray/dict of latents, got {type(obj)}"
        )

    if len(obj) != len(seeds):
        raise ValueError(
            f"{path}: contains {len(obj)} latent entries, "
            f"but {len(seeds)} seeds were provided"
        )

    out = []
    for seed, z in zip(seeds, obj):
        z = np.asarray(z)

        if z.ndim == 2:
            z = z[:, None, :]

        if z.ndim != 3:
            raise ValueError(
                f"{path}, seed {seed}: expected (n_cells, n_gfm, d), got {z.shape}"
            )

        out.append(clean_array(z))

    return out


def build_mat2(latent_val, n_neighbour, red_local):
    n_sample, n_gfm, _ = latent_val.shape

    if n_neighbour >= n_sample:
        raise ValueError(f"n_neighbour={n_neighbour} must be < n_cells={n_sample}")

    dist = np.zeros(shape=(n_gfm, n_sample, n_sample))

    for c in range(n_gfm):
        z = reducer(red_local)(clean_array(latent_val[:, c, :]))

        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                dist[c, i, j] = dist[c, j, i] = np.linalg.norm(z[i] - z[j])

    neighbour_dist = np.array(
        [
            np.array(
                [
                    np.sum(
                        dis[
                            i,
                            np.argpartition(dis[i], n_neighbour + 1)[
                                1 : n_neighbour + 1
                            ],
                        ]
                    )
                    / n_neighbour
                    for i in range(n_sample)
                ]
            )
            for dis in dist
        ]
    )

    neighbour_dist_score = (
        np.array(
            [
                np.array(
                    [
                        1
                        / np.log(
                            np.var(
                                dis[
                                    i,
                                    np.argpartition(dis[i], n_neighbour + 1)[
                                        1 : n_neighbour + 1
                                    ],
                                ]
                            )
                            + np.exp(1)
                        )
                        for i in range(n_sample)
                    ]
                )
                for dis in dist
            ]
        )
        ** 0.5
    )

    return get_mat2(n_sample, neighbour_dist, dist, neighbour_dist_score)


def build_mat1(latent_val, n_clusters, kmeans_times, red_global, thread_num):
    n_sample, n_gfm, _ = latent_val.shape

    pred = np.zeros(shape=(kmeans_times, n_gfm, n_sample)).astype(int)
    score = np.zeros(shape=(kmeans_times, n_gfm, n_sample))

    for c in range(n_gfm):
        z = clean_array(latent_val[:, c, :]).reshape(n_sample, -1)
        z = reducer(red_global)(z)

        for t in range(kmeans_times):
            pred_z, score_z = c_kmeans(
                z,
                n_clusters,
                n_init=10,
                random_state=None,
            )

            pred[t, c, :] = pred_z
            score_z = np.sort(score_z, axis=1)

            denom = score_z[:, 1] + score_z[:, 0]
            denom = np.where(denom == 0, 1e-12, denom)

            ratio = (score_z[:, 1] - score_z[:, 0]) / denom

            score[t, c, :] = ratio**0.5 / kmeans_times / n_gfm

    mat1 = get_mat1(
        pred,
        n_sample,
        kmeans_times,
        n_gfm,
        n_clusters,
        score,
        thread_num,
    )

    mean_mat1 = np.mean(mat1)
    if mean_mat1 > 1e-12:
        mat1 = mat1 / mean_mat1

    return mat1


def kmeans_concat(latent_val, n_clusters, seed):
    z = latent_val.reshape(latent_val.shape[0], -1)

    return KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
    ).fit_predict(clean_array(z))


def spectral_knn_concat(latent_val, n_clusters, knn, seed):
    z = latent_val.reshape(latent_val.shape[0], -1)

    if knn >= z.shape[0]:
        raise ValueError(f"knn={knn} must be < n_cells={z.shape[0]}")

    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=knn,
        assign_labels="kmeans",
        random_state=seed,
    ).fit_predict(clean_array(z))


def spectral_precomputed(mat, n_clusters, seed):
    mat = clean_array(mat)
    mat = (mat + mat.T) / 2.0
    mat = np.clip(mat, 0.0, None)

    return SpectralClustering(
        n_clusters=n_clusters,
        random_state=seed,
        affinity="precomputed",
        assign_labels="kmeans",
    ).fit_predict(mat)


def evaluate_arm(arm_name, latents, seeds, y, args, alpha_beta_pairs):
    rows = []

    for seed, latent_val in zip(seeds, latents):
        if latent_val.shape[0] != len(y):
            raise ValueError(
                f"{arm_name}, seed {seed}: "
                f"n_cells={latent_val.shape[0]} but len(y)={len(y)}"
            )

        print(f"[{arm_name}] seed={seed}: direct baselines")

        baseline_methods = [
            ("B_kmeans_concat", kmeans_concat(latent_val, args.cluster_number, seed)),
            (
                "B_spectral_knn_concat",
                spectral_knn_concat(
                    latent_val,
                    args.cluster_number,
                    args.knn,
                    seed,
                ),
            ),
        ]

        for method, labels in baseline_methods:
            nmi, ari, acc = metrics(y, labels)
            rows.append((arm_name, seed, -1, method, np.nan, np.nan, nmi, ari, acc))

        print(f"[{arm_name}] seed={seed}: building mat2/s_d")
        mat2 = build_mat2(latent_val, args.n_neighbour, args.red_local)

        for repeat_idx in range(args.repeat):
            print(f"[{arm_name}] seed={seed}: repeat={repeat_idx}, building mat1/s_g")

            set_seed(seed + repeat_idx)

            mat1 = build_mat1(
                latent_val,
                n_clusters=args.cluster_number,
                kmeans_times=args.kmeans_times,
                red_global=args.red_global,
                thread_num=args.thread_num,
            )

            for alpha, beta in alpha_beta_pairs:
                mat = mat1 * alpha + mat2 * beta

                labels = spectral_precomputed(
                    mat,
                    args.cluster_number,
                    seed + repeat_idx,
                )

                nmi, ari, acc = metrics(y, labels)

                if alpha == 0 and beta == 1:
                    method = "D_spectral_mat2_only"
                elif alpha == 1 and beta == 0:
                    method = "C_spectral_mat1_only"
                else:
                    method = "CD_spectral_alpha_beta"

                rows.append(
                    (arm_name, seed, repeat_idx, method, alpha, beta, nmi, ari, acc)
                )

    return rows


def summarise(rows):
    groups = defaultdict(list)

    for arm, seed, repeat_idx, method, alpha, beta, nmi, ari, acc in rows:
        key = (arm, method, alpha, beta)
        groups[key].append((nmi, ari, acc))

    summary = []

    for key, vals in sorted(groups.items(), key=lambda x: str(x[0])):
        arr = np.asarray(vals, dtype=float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        summary.append((*key, len(vals), *mean, *std))

    return summary


def oracle_best_by_seed(rows, arm, method_prefix="CD_spectral"):
    by_seed = defaultdict(list)

    for row in rows:
        r_arm, seed, repeat_idx, method, alpha, beta, nmi, ari, acc = row

        if r_arm == arm and method.startswith(method_prefix):
            by_seed[seed].append((nmi, ari, acc))

    out = {}

    for seed, vals in by_seed.items():
        arr = np.asarray(vals, dtype=float)
        out[seed] = tuple(arr[np.argmax(arr[:, 0])])

    return out


def method_mean_by_seed(rows, arm, method_name):
    by_seed = defaultdict(list)

    for row in rows:
        r_arm, seed, repeat_idx, method, alpha, beta, nmi, ari, acc = row

        if r_arm == arm and method == method_name:
            by_seed[seed].append((nmi, ari, acc))

    return {
        seed: tuple(np.asarray(vals, dtype=float).mean(axis=0))
        for seed, vals in by_seed.items()
    }


def paired_wilcoxon(a, b, seeds):
    out = []

    A = np.asarray([a[s] for s in seeds], dtype=float)
    B = np.asarray([b[s] for s in seeds], dtype=float)

    for i, metric in enumerate(["NMI", "ARI", "ACC"]):
        diff = A[:, i] - B[:, i]

        try:
            _, p = wilcoxon(diff)
        except Exception:
            p = np.nan

        out.append(
            (
                metric,
                diff.mean(),
                int((diff > 0).sum()),
                int((diff < 0).sum()),
                int((diff == 0).sum()),
                p,
            )
        )

    return out


def write_outputs(rows, outfile):
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

    summary_file = outfile.replace(".tsv", "_summary.tsv")

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("arm\tseed\trepeat\tmethod\talpha\tbeta\tnmi\tari\tacc\n")

        for row in rows:
            arm, seed, repeat_idx, method, alpha, beta, nmi, ari, acc = row

            f.write(
                f"{arm}\t{seed}\t{repeat_idx}\t{method}\t"
                f"{alpha}\t{beta}\t{nmi:.6f}\t{ari:.6f}\t{acc:.6f}\n"
            )

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            "arm\tmethod\talpha\tbeta\tn\t"
            "nmi_mean\tari_mean\tacc_mean\t"
            "nmi_std\tari_std\tacc_std\n"
        )

        for row in summarise(rows):
            arm, method, alpha, beta, n, nmi_m, ari_m, acc_m, nmi_s, ari_s, acc_s = row

            f.write(
                f"{arm}\t{method}\t{alpha}\t{beta}\t{n}\t"
                f"{nmi_m:.6f}\t{ari_m:.6f}\t{acc_m:.6f}\t"
                f"{nmi_s:.6f}\t{ari_s:.6f}\t{acc_s:.6f}\n"
            )

    print(f"Wrote per-run results: {outfile}")
    print(f"Wrote summary results: {summary_file}")


def parse_alpha_beta(s):
    if not s:
        return DEFAULT_ALPHA_BETA

    pairs = []

    for item in s.split(","):
        a, b = item.split(":")
        pairs.append((float(a), float(b)))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", default="muraro", type=str)

    parser.add_argument(
        "--cluster_number",
        "--cluster-number",
        dest="cluster_number",
        default=None,
        type=int,
    )

    parser.add_argument("--seeds", default=DEFAULT_SEEDS, type=str)

    parser.add_argument(
        "--latents-autoencoder",
        default="./outputs/muraro_latents_autoencoder.joblib",
    )

    parser.add_argument(
        "--latents-dmkcn",
        default="./outputs/muraro_latents_dmkcn.joblib",
    )

    parser.add_argument(
        "--outfile",
        default="./outputs/ablation_block_c_d_muraro.tsv",
    )

    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--n_neighbour", default=3, type=int)
    parser.add_argument("--kmeans_times", default=20, type=int)

    parser.add_argument("--red_global", type=str, default=None)
    parser.add_argument("--red_local", type=str, default=None)

    parser.add_argument("--thread-num", default=8, type=int)

    parser.add_argument(
        "--knn",
        default=15,
        type=int,
        help="kNN used only for the direct spectral baseline.",
    )

    parser.add_argument(
        "--alpha-beta",
        default=None,
        type=str,
        help="Optional comma-separated alpha:beta list. Default is scMUG's 9 fixed pairs.",
    )

    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    alpha_beta_pairs = parse_alpha_beta(args.alpha_beta)

    expr_df, cell_type = load_data(args.dataset)

    adata = preprocess(
        expr_df=expr_df.astype(float),
        cell_type=cell_type,
        highly_genes=8000,
    )

    y = lab2fac(adata.obs["cell_type"].to_numpy())

    if args.cluster_number is None:
        args.cluster_number = len(set(y))

    print(
        f"Dataset: {args.dataset}; n_cells={len(y)}; n_clusters={args.cluster_number}"
    )
    print(f"Alpha/beta pairs: {alpha_beta_pairs}")

    latents_auto = load_latent_list(args.latents_autoencoder, seeds)
    latents_dmkcn = load_latent_list(args.latents_dmkcn, seeds)

    rows = []

    rows.extend(
        evaluate_arm(
            "autoencoder",
            latents_auto,
            seeds,
            y,
            args,
            alpha_beta_pairs,
        )
    )

    rows.extend(
        evaluate_arm(
            "dmkcn",
            latents_dmkcn,
            seeds,
            y,
            args,
            alpha_beta_pairs,
        )
    )

    write_outputs(rows, args.outfile)

    print("\nPaired sanity checks, delta = first - second")

    for arm in ["autoencoder", "dmkcn"]:
        best_cd = oracle_best_by_seed(rows, arm)
        km = method_mean_by_seed(rows, arm, "B_kmeans_concat")

        if set(best_cd) == set(km) == set(seeds):
            print(f"\n{arm}: oracle best C/D vs B_kmeans_concat")

            for metric, delta, wins, losses, ties, p in paired_wilcoxon(
                best_cd,
                km,
                seeds,
            ):
                print(
                    f"  {metric}: delta={delta:+.4f}; "
                    f"wins={wins}; losses={losses}; ties={ties}; p={p:.4g}"
                )

    best_auto = oracle_best_by_seed(rows, "autoencoder")
    best_dmkcn = oracle_best_by_seed(rows, "dmkcn")

    if set(best_auto) == set(best_dmkcn) == set(seeds):
        print("\ndmkcn oracle best C/D vs autoencoder oracle best C/D")

        for metric, delta, wins, losses, ties, p in paired_wilcoxon(
            best_dmkcn,
            best_auto,
            seeds,
        ):
            print(
                f"  {metric}: delta={delta:+.4f}; "
                f"wins={wins}; losses={losses}; ties={ties}; p={p:.4g}"
            )


if __name__ == "__main__":
    main()
