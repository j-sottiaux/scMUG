"""Ablate scMUG downstream blocks C and D from saved block-B latents.

Definitions used here:
  - block C: construction of the scMUG cell-cell affinity from block-B latents
             (global mat1 + local-density mat2).
  - block D: final spectral clustering on the block-C affinity.

The previous labels "no_C = mat2 only" and "no_D = mat1 only" confounded C's
internal components with the downstream clustering block. This script separates
those axes:
  - C_full_D_spectral: recomputed full downstream path, for QC.
  - C_global_only_D_spectral: C local-density component removed.
  - C_local_only_D_spectral: C global-distribution component removed.
  - no_C_D_spectral_C_input_knn: C removed, D kept as spectral clustering on a
                                 direct kNN affinity built from the same reduced
                                 per-GFM representations consumed by C.
  - C_full_no_D_kmeans_rows: C kept, D replaced by k-means on affinity rows.
  - no_C_no_D_kmeans_C_input: C and D removed; k-means on the same reduced
                              per-GFM representations consumed by C.

Full scMUG / full scMUG-DMKCN should still be obtained from scMUG.py; the
recomputed full arm here is a paired sanity check using the saved latents.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import joblib
import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from accelerate import get_mat1, get_mat1_job, get_mat2
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


def parse_alpha_beta_grid(value):
    """Parse ``alpha:beta`` pairs while preserving order and removing duplicates."""
    if value is None:
        return None

    pairs = []
    seen = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "alpha/beta grid entries must use 'alpha:beta', for example "
                "'0.01:1,0.1:1,1:1'"
            )
        try:
            alpha, beta = (float(field.strip()) for field in fields)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid alpha/beta pair '{item}': expected numeric values"
            ) from exc
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise argparse.ArgumentTypeError(
                f"invalid alpha/beta pair '{item}': values must be finite"
            )
        if alpha < 0 or beta < 0 or (alpha == 0 and beta == 0):
            raise argparse.ArgumentTypeError(
                f"invalid alpha/beta pair '{item}': weights must be non-negative "
                "and cannot both be zero"
            )
        pair = (alpha, beta)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    if not pairs:
        raise argparse.ArgumentTypeError("alpha/beta grid must contain at least one pair")
    return pairs


def full_cd_grid_conditions(alpha_beta_pairs):
    """Return the two block-D conditions evaluated for every C-weight pair."""
    conditions = []
    for alpha, beta in alpha_beta_pairs:
        conditions.extend(
            [
                (
                    "C_full_D_spectral",
                    alpha,
                    beta,
                    "full",
                    "spectral_precomputed",
                ),
                (
                    "C_full_no_D_kmeans_rows",
                    alpha,
                    beta,
                    "full",
                    "kmeans_affinity_rows",
                ),
            ]
        )
    return conditions


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
        raise TypeError(f"{path}: expected list/tuple/ndarray/dict, got {type(obj)}")

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

    dist = np.zeros((n_gfm, n_sample, n_sample))

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


def build_mat1(latent_val, n_clusters, kmeans_times, red_global, thread_num, seed):
    n_sample, n_gfm, _ = latent_val.shape

    pred = np.zeros((kmeans_times, n_gfm, n_sample)).astype(int)
    score = np.zeros((kmeans_times, n_gfm, n_sample))

    for c in range(n_gfm):
        z = clean_array(latent_val[:, c, :]).reshape(n_sample, -1)
        z = reducer(red_global)(z)

        for t in range(kmeans_times):
            kmeans_seed = seed + c * kmeans_times + t
            pred_z, score_z = c_kmeans(
                z,
                n_clusters,
                n_init=10,
                random_state=kmeans_seed,
            )

            pred[t, c, :] = pred_z
            score_z = np.sort(score_z, axis=1)

            denom = score_z[:, 1] + score_z[:, 0]
            denom = np.where(denom == 0, 1e-12, denom)

            ratio = (score_z[:, 1] - score_z[:, 0]) / denom
            score[t, c, :] = ratio**0.5 / kmeans_times / n_gfm

    if thread_num <= 1:
        mat1 = get_mat1_job(
            pred,
            n_sample,
            kmeans_times,
            n_gfm,
            n_clusters,
            score,
            1,
            0,
        )
    else:
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


def concat_block_c_inputs(latent_val, red_global, red_local):
    """Return cells x features exposed to block C, concatenated across GFMs.

    scMUG's block C does not consume the high-dimensional B latents directly: it
    reduces each GFM separately before building mat1 and mat2. For a fair C+D
    ablation, direct clustering uses those same reduced per-GFM views. If the
    global and local reducers differ, both views are concatenated.
    """
    n_sample, n_gfm, _ = latent_val.shape
    features = []

    reducer_names = [red_global]
    if red_local != red_global:
        reducer_names.append(red_local)

    for reducer_name in reducer_names:
        red = reducer(reducer_name)
        for c in range(n_gfm):
            z = clean_array(latent_val[:, c, :]).reshape(n_sample, -1)
            features.append(clean_array(red(z)))

    return StandardScaler().fit_transform(np.concatenate(features, axis=1))


def direct_knn_affinity(features, knn):
    """Direct affinity from block-C input features, used when C is removed."""
    if knn >= features.shape[0]:
        raise ValueError(f"direct knn={knn} must be < n_cells={features.shape[0]}")

    graph = kneighbors_graph(
        features,
        n_neighbors=knn,
        mode="connectivity",
        include_self=False,
    )
    graph = graph.maximum(graph.T)
    affinity = graph.toarray() if sparse.issparse(graph) else np.asarray(graph)
    np.fill_diagonal(affinity, 1.0)
    return affinity.astype(float)


def spectral_precomputed(mat, n_clusters, seed):
    mat = clean_array(mat)
    mat = (mat + mat.T) / 2.0
    mat = np.clip(mat, 0.0, None)

    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    ).fit_predict(mat)


def kmeans_features(x, n_clusters, seed, n_init):
    x = clean_array(x)
    x = StandardScaler().fit_transform(x)
    labels, _ = c_kmeans(
        x,
        n_clusters,
        n_init=n_init,
        random_state=seed,
    )
    return labels


def kmeans_affinity_rows(mat, n_clusters, seed, n_init):
    """Replacement for block D: cluster cells by their C-affinity profiles."""
    mat = clean_array(mat)
    mat = (mat + mat.T) / 2.0
    mat = np.clip(mat, 0.0, None)
    return kmeans_features(mat, n_clusters, seed, n_init)


def append_result(
    rows,
    arm_name,
    seed,
    repeat_idx,
    method,
    alpha,
    beta,
    c_mode,
    d_mode,
    red_global,
    red_local,
    y,
    labels,
):
    nmi, ari, acc = metrics(y, labels)
    rows.append(
        (
            arm_name,
            seed,
            repeat_idx,
            method,
            alpha,
            beta,
            c_mode,
            d_mode,
            red_global,
            red_local,
            nmi,
            ari,
            acc,
        )
    )


def evaluate_arm(arm_name, latents, seeds, y, args):
    rows = []
    grid_conditions = (
        full_cd_grid_conditions(args.alpha_beta_grid)
        if args.alpha_beta_grid is not None
        else None
    )

    for seed, latent_val in zip(seeds, latents):
        if latent_val.shape[0] != len(y):
            raise ValueError(
                f"{arm_name}, seed {seed}: "
                f"n_cells={latent_val.shape[0]} but len(y)={len(y)}"
            )

        print(f"[{arm_name}] seed={seed}: building C-local mat2")
        mat2 = build_mat2(latent_val, args.n_neighbour, args.red_local)
        if grid_conditions is None:
            direct_features = concat_block_c_inputs(
                latent_val,
                args.red_global,
                args.red_local,
            )
            direct_affinity = direct_knn_affinity(direct_features, args.direct_knn)

        for repeat_idx in range(args.repeat):
            print(
                f"[{arm_name}] seed={seed}: repeat={repeat_idx}, building C-global mat1"
            )

            set_seed(seed + repeat_idx)

            mat1 = build_mat1(
                latent_val,
                n_clusters=args.cluster_number,
                kmeans_times=args.kmeans_times,
                red_global=args.red_global,
                thread_num=args.thread_num,
                seed=seed + repeat_idx,
            )

            if grid_conditions is not None:
                mat_by_pair = {
                    (alpha, beta): alpha * mat1 + beta * mat2
                    for alpha, beta in args.alpha_beta_grid
                }
                for method, alpha, beta, c_mode, d_mode in grid_conditions:
                    mat_full = mat_by_pair[(alpha, beta)]
                    if d_mode == "spectral_precomputed":
                        labels = spectral_precomputed(
                            mat_full,
                            args.cluster_number,
                            seed + repeat_idx,
                        )
                    elif d_mode == "kmeans_affinity_rows":
                        labels = kmeans_affinity_rows(
                            mat_full,
                            args.cluster_number,
                            seed + repeat_idx,
                            args.direct_kmeans_n_init,
                        )
                    else:  # pragma: no cover - guarded by full_cd_grid_conditions
                        raise ValueError(f"Unknown grid D mode: {d_mode}")

                    append_result(
                        rows,
                        arm_name,
                        seed,
                        repeat_idx,
                        method,
                        alpha,
                        beta,
                        c_mode,
                        d_mode,
                        args.red_global,
                        args.red_local,
                        y,
                        labels,
                    )
                continue

            mat_full = args.full_alpha * mat1 + args.full_beta * mat2
            protocol = [
                (
                    "C_full_D_spectral",
                    args.full_alpha,
                    args.full_beta,
                    "full",
                    "spectral_precomputed",
                    lambda s: spectral_precomputed(mat_full, args.cluster_number, s),
                ),
                (
                    "C_global_only_D_spectral",
                    1.0,
                    0.0,
                    "global_only",
                    "spectral_precomputed",
                    lambda s: spectral_precomputed(mat1, args.cluster_number, s),
                ),
                (
                    "C_local_only_D_spectral",
                    0.0,
                    1.0,
                    "local_only",
                    "spectral_precomputed",
                    lambda s: spectral_precomputed(mat2, args.cluster_number, s),
                ),
                (
                    "no_C_D_spectral_C_input_knn",
                    np.nan,
                    np.nan,
                    "direct_knn_from_C_input",
                    "spectral_precomputed",
                    lambda s: spectral_precomputed(
                        direct_affinity,
                        args.cluster_number,
                        s,
                    ),
                ),
                (
                    "C_full_no_D_kmeans_rows",
                    args.full_alpha,
                    args.full_beta,
                    "full",
                    "kmeans_affinity_rows",
                    lambda s: kmeans_affinity_rows(
                        mat_full,
                        args.cluster_number,
                        s,
                        args.direct_kmeans_n_init,
                    ),
                ),
                (
                    "no_C_no_D_kmeans_C_input",
                    np.nan,
                    np.nan,
                    "direct_C_input",
                    "kmeans_features",
                    lambda s: kmeans_features(
                        direct_features,
                        args.cluster_number,
                        s,
                        args.direct_kmeans_n_init,
                    ),
                ),
            ]

            for method, alpha, beta, c_mode, d_mode, runner in protocol:
                labels = runner(seed + repeat_idx)
                append_result(
                    rows,
                    arm_name,
                    seed,
                    repeat_idx,
                    method,
                    alpha,
                    beta,
                    c_mode,
                    d_mode,
                    args.red_global,
                    args.red_local,
                    y,
                    labels,
                )

    return rows


def summarise(rows):
    groups = defaultdict(list)

    for row in rows:
        (
            arm,
            seed,
            repeat_idx,
            method,
            alpha,
            beta,
            c_mode,
            d_mode,
            red_global,
            red_local,
            nmi,
            ari,
            acc,
        ) = row
        key = (arm, method, alpha, beta, c_mode, d_mode, red_global, red_local)
        groups[key].append((nmi, ari, acc))

    summary = []

    for key, vals in sorted(groups.items(), key=lambda x: str(x[0])):
        arr = np.asarray(vals, dtype=float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        minv = arr.min(axis=0)
        maxv = arr.max(axis=0)

        summary.append((*key, len(vals), *mean, *std, *minv, *maxv))

    return summary


def write_outputs(rows, outfile):
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)

    summary_file = outfile.replace(".tsv", "_summary.tsv")

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(
            "arm\tseed\trepeat\tmethod\talpha\tbeta\tc_mode\td_mode\t"
            "red_global\tred_local\tnmi\tari\tacc\n"
        )

        for row in rows:
            (
                arm,
                seed,
                repeat_idx,
                method,
                alpha,
                beta,
                c_mode,
                d_mode,
                red_global,
                red_local,
                nmi,
                ari,
                acc,
            ) = row

            f.write(
                f"{arm}\t{seed}\t{repeat_idx}\t{method}\t"
                f"{alpha}\t{beta}\t{c_mode}\t{d_mode}\t"
                f"{red_global}\t{red_local}\t"
                f"{nmi:.6f}\t{ari:.6f}\t{acc:.6f}\n"
            )

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            "arm\tmethod\talpha\tbeta\tc_mode\td_mode\tred_global\tred_local\tn\t"
            "nmi_mean\tari_mean\tacc_mean\t"
            "nmi_std\tari_std\tacc_std\t"
            "nmi_min\tari_min\tacc_min\t"
            "nmi_max\tari_max\tacc_max\n"
        )

        for row in summarise(rows):
            (
                arm,
                method,
                alpha,
                beta,
                c_mode,
                d_mode,
                red_global,
                red_local,
                n,
                nmi_m,
                ari_m,
                acc_m,
                nmi_s,
                ari_s,
                acc_s,
                nmi_min,
                ari_min,
                acc_min,
                nmi_max,
                ari_max,
                acc_max,
            ) = row

            f.write(
                f"{arm}\t{method}\t{alpha}\t{beta}\t"
                f"{c_mode}\t{d_mode}\t{red_global}\t{red_local}\t{n}\t"
                f"{nmi_m:.6f}\t{ari_m:.6f}\t{acc_m:.6f}\t"
                f"{nmi_s:.6f}\t{ari_s:.6f}\t{acc_s:.6f}\t"
                f"{nmi_min:.6f}\t{ari_min:.6f}\t{acc_min:.6f}\t"
                f"{nmi_max:.6f}\t{ari_max:.6f}\t{acc_max:.6f}\n"
            )

    print(f"Wrote per-run results: {outfile}")
    print(f"Wrote summary results: {summary_file}")


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
        "--dmkcn-manifest",
        default=None,
        help="Optional scMUG DMKCN manifest; evaluates every saved K projection.",
    )

    parser.add_argument(
        "--outfile",
        default="./outputs/ablation_noC_noD_noCD_muraro.tsv",
    )

    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--n_neighbour", default=3, type=int)
    parser.add_argument("--kmeans_times", default=20, type=int)

    parser.add_argument("--red_global", type=str, default="umap")
    parser.add_argument("--red_local", type=str, default="umap")

    parser.add_argument("--thread-num", default=8, type=int)

    parser.add_argument(
        "--full-alpha",
        default=1.0,
        type=float,
        help="Canonical alpha used for full C = alpha * mat1 + beta * mat2.",
    )

    parser.add_argument(
        "--full-beta",
        default=1.0,
        type=float,
        help="Canonical beta used for full C = alpha * mat1 + beta * mat2.",
    )

    parser.add_argument(
        "--alpha-beta-grid",
        type=parse_alpha_beta_grid,
        default=None,
        help=(
            "Optional comma-separated alpha:beta pairs. When provided, evaluate "
            "only full C crossed with spectral clustering and k-means on affinity "
            "rows, without running the other C/D ablations."
        ),
    )

    parser.add_argument(
        "--direct-knn",
        "--knn",
        dest="direct_knn",
        default=15,
        type=int,
        help="kNN used to build the direct B-latent affinity when block C is removed.",
    )

    parser.add_argument(
        "--direct-kmeans-n-init",
        default=20,
        type=int,
        help="n_init used by direct k-means baselines and D replacement.",
    )

    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]

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

    if args.alpha_beta_grid is not None:
        print(
            "C/D factorial grid: C_full_D_spectral x "
            "C_full_no_D_kmeans_rows; "
            f"alpha_beta_grid={args.alpha_beta_grid}; "
            f"red_global={args.red_global}; red_local={args.red_local}"
        )
    else:
        print(
            "Ablations: C_full_D_spectral, C_global_only_D_spectral, "
            "C_local_only_D_spectral, no_C_D_spectral_C_input_knn, "
            "C_full_no_D_kmeans_rows, no_C_no_D_kmeans_C_input; "
            f"full_alpha={args.full_alpha}; full_beta={args.full_beta}; "
            f"red_global={args.red_global}; red_local={args.red_local}"
        )

    latents_auto = load_latent_list(args.latents_autoencoder, seeds)
    rows = []

    rows.extend(
        evaluate_arm(
            "autoencoder",
            latents_auto,
            seeds,
            y,
            args,
        )
    )

    if args.dmkcn_manifest:
        with open(args.dmkcn_manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        variant_paths = manifest.get("variant_paths", {})
        if not variant_paths:
            raise ValueError(
                f"DMKCN manifest has no variant_paths: {args.dmkcn_manifest}"
            )
        for variant_key, variant_path in sorted(variant_paths.items()):
            print(f"Evaluating DMKCN latent variant: {variant_key}")
            rows.extend(
                evaluate_arm(
                    f"dmkcn__{variant_key}",
                    load_latent_list(variant_path, seeds),
                    seeds,
                    y,
                    args,
                )
            )
    else:
        latents_dmkcn = load_latent_list(args.latents_dmkcn, seeds)
        rows.extend(
            evaluate_arm(
                "dmkcn",
                latents_dmkcn,
                seeds,
                y,
                args,
            )
        )

    write_outputs(rows, args.outfile)


if __name__ == "__main__":
    main()
