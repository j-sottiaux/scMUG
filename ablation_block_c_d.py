"""Ablation of scMUG blocks C+D as a 2x2 factorial design.

Questions:
  Q1: Does scMUG block C/D help, given a fixed block-B representation?
  Q2: Is the scDMKC representation better than the original scMUG autoencoder representation?

Design:

                       | COMPLETE: block C/D | direct spectral: no C/D |
    scMUG latent B     |        (1)          |          (2)            |
    scDMKC latent B    |        (3)          |          (4)            |

Inputs:
  - COMPLETE cells are read from scMUG-format result .txt files.
  - Direct cells are computed here from saved latent dumps:
        concat per-GFM embeddings -> kNN spectral clustering.

Important caveats:
  - COMPLETE oracle-best selects alpha/beta using labels; optimistic.
  - COMPLETE all-configs mean is less optimistic but mixes good and bad alpha/beta values.
  - Direct spectral is untuned: fixed kNN, no alpha/beta sweep.
  - scDMKC may also differ by ZINB target choice; isolate separately if needed.

Example:
    python ablation_block_c_d.py \
        --latents-scmug ./outputs/muraro_latents_scmug.joblib \
        --complete-scmug ./outputs/muraros_scmug.txt \
        --latents-dmkcn ./outputs/muraro_latents_dmkcn.joblib \
        --complete-dmkcn ./outputs/muraros_scmug_dmkcn.txt \
        --outfile ./outputs/ablation_block_c_d_summary.tsv
"""

import argparse
import os
import re
from collections import defaultdict

import joblib
import numpy as np
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans, SpectralClustering

from utils import calc_acc, calc_ari, calc_nmi, lab2fac, load_data, preprocess


DEFAULT_SEEDS = "1111,2222,3333,4444,5555,6666,7777,8888,9999,10000"


def metrics(y, labels):
    """Return metrics in fixed order: (NMI, ARI, ACC)."""
    return calc_nmi(y, labels), calc_ari(y, labels), calc_acc(y, labels)


def clean_Z(Z):
    """Convert an embedding to a finite float array."""
    return np.nan_to_num(
        np.asarray(Z, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def spectral_direct(Z, n_clusters, knn, seed):
    """Spectral clustering on a kNN graph of Z, without scMUG s_g/s_d construction."""
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


def kmeans_direct(Z, n_clusters, seed):
    """K-means directly on concatenated latent space. Diagnostic baseline only."""
    return KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
    ).fit_predict(clean_Z(Z))


def parse_complete(path):
    """Parse a scMUG-format result file.

    Returns:
        best_by_seed:
            dict seed -> oracle-best row selected by max NMI.
        mean_by_seed:
            dict seed -> mean over all alpha/beta/repeat rows for that seed.
        allrows:
            np.ndarray of all rows.

    Row order is always:
        (NMI, ARI, ACC)
    """
    byseed = defaultdict(list)

    pat = re.compile(
        r"round:(\d+).*?acc:\s*([\d.]+)\s*\tari:\s*([\d.]+)\s*\tnmi:\s*([\d.]+)"
    )

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue

            seed, acc, ari, nmi = m.groups()
            byseed[int(seed)].append((float(nmi), float(ari), float(acc)))

    if not byseed:
        raise ValueError(f"No scMUG-format rows found in {path}")

    best_by_seed = {}
    mean_by_seed = {}

    for seed, rows in byseed.items():
        rows_arr = np.asarray(rows, dtype=float)
        best_by_seed[seed] = tuple(rows_arr[np.argmax(rows_arr[:, 0])])
        mean_by_seed[seed] = tuple(rows_arr.mean(axis=0))

    allrows = np.asarray([r for rows in byseed.values() for r in rows], dtype=float)

    return best_by_seed, mean_by_seed, allrows


def require_seed_coverage(name, d, seeds):
    missing = sorted(set(seeds) - set(d))
    extra = sorted(set(d) - set(seeds))

    if missing:
        raise ValueError(f"{name}: missing seeds {missing}")

    if extra:
        print(f"[warning] {name}: ignoring extra seeds {extra}")

    return {s: d[s] for s in seeds}


def run_direct(latents_path, seeds, y, n_clusters, knn):
    """Direct spectral and k-means on concatenated GFM embeddings, per seed.

    Returns:
        spec: dict seed -> (NMI, ARI, ACC)
        km:   dict seed -> (NMI, ARI, ACC)
    """
    latents = joblib.load(latents_path)

    if len(seeds) != len(latents):
        raise ValueError(
            f"{latents_path}: {len(latents)} latent dumps but {len(seeds)} seeds"
        )

    spec = {}
    km = {}

    for si, lv in enumerate(latents):
        seed = seeds[si]
        lv = np.asarray(lv)

        if lv.ndim != 3:
            raise ValueError(
                f"{latents_path}, seed {seed}: expected shape "
                f"(n_cells, n_gfm, d), got {lv.shape}"
            )

        n_cells = lv.shape[0]

        if len(y) != n_cells:
            raise ValueError(
                f"{latents_path}, seed {seed}: labels length {len(y)} "
                f"!= n_cells {n_cells}"
            )

        Z = lv.reshape(n_cells, -1)

        spec[seed] = metrics(
            y,
            spectral_direct(Z, n_clusters=n_clusters, knn=knn, seed=seed),
        )

        km[seed] = metrics(
            y,
            kmeans_direct(Z, n_clusters=n_clusters, seed=seed),
        )

    return spec, km


def dict_mean_std(d):
    a = np.asarray([d[s] for s in sorted(d)], dtype=float)
    return a.mean(axis=0), a.std(axis=0)


def print_line(name, mean, std=None):
    if std is None:
        print(
            f"  {name:<38} "
            f"NMI {mean[0]:.4f}       "
            f"ARI {mean[1]:.4f}       "
            f"ACC {mean[2]:.4f}"
        )
    else:
        print(
            f"  {name:<38} "
            f"NMI {mean[0]:.4f}±{std[0]:.3f}  "
            f"ARI {mean[1]:.4f}±{std[1]:.3f}  "
            f"ACC {mean[2]:.4f}±{std[2]:.3f}"
        )


def paired(a_dict, b_dict, label, seeds):
    """Paired Wilcoxon tests. Delta = a - b."""
    A = np.asarray([a_dict[s] for s in seeds], dtype=float)
    B = np.asarray([b_dict[s] for s in seeds], dtype=float)

    parts = []

    for i, metric_name in enumerate(["NMI", "ARI", "ACC"]):
        diff = A[:, i] - B[:, i]
        wins = int(np.sum(diff > 0))
        losses = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))

        try:
            _, p = wilcoxon(diff)
        except Exception:
            p = float("nan")

        parts.append(
            f"{metric_name} Δ={diff.mean():+.4f} "
            f"wins={wins}/{len(diff)} losses={losses} ties={ties} p={p:.3f}"
        )

    print(f"  {label:<38} " + " | ".join(parts))


def write_method_rows(f, arm, method, d, seeds):
    """Write per-seed method results to TSV."""
    for seed in seeds:
        nmi, ari, acc = d[seed]
        f.write(f"{arm}\t{method}\t{seed}\t{nmi:.6f}\t{ari:.6f}\t{acc:.6f}\n")


def write_paired_rows(f, comparison, a_dict, b_dict, seeds):
    """Write paired deltas to TSV."""
    A = np.asarray([a_dict[s] for s in seeds], dtype=float)
    B = np.asarray([b_dict[s] for s in seeds], dtype=float)

    for i, metric_name in enumerate(["NMI", "ARI", "ACC"]):
        diff = A[:, i] - B[:, i]

        try:
            _, p = wilcoxon(diff)
        except Exception:
            p = float("nan")

        wins = int(np.sum(diff > 0))
        losses = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))

        f.write(
            f"{comparison}\t{metric_name}\t"
            f"{diff.mean():.6f}\t{diff.std():.6f}\t"
            f"{wins}\t{losses}\t{ties}\t{p:.6g}\n"
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="muraro")
    ap.add_argument("--latents-scmug", required=True)
    ap.add_argument("--complete-scmug", required=True)
    ap.add_argument("--latents-dmkcn", required=True)
    ap.add_argument("--complete-dmkcn", required=True)

    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--outfile", default="./outputs/ablation_block_c_d_summary.tsv")

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

    print(f"{args.dataset}: {n_clusters} types | knn={args.knn} | seeds={len(seeds)}\n")

    # Direct cells: (2) and (4)
    spec_s, km_s = run_direct(
        args.latents_scmug,
        seeds=seeds,
        y=y,
        n_clusters=n_clusters,
        knn=args.knn,
    )

    spec_d, km_d = run_direct(
        args.latents_dmkcn,
        seeds=seeds,
        y=y,
        n_clusters=n_clusters,
        knn=args.knn,
    )

    # Complete cells: (1) and (3)
    best_s, mean_s, all_s = parse_complete(args.complete_scmug)
    best_d, mean_d, all_d = parse_complete(args.complete_dmkcn)

    best_s = require_seed_coverage("complete-scmug oracle-best", best_s, seeds)
    mean_s = require_seed_coverage("complete-scmug per-seed mean", mean_s, seeds)
    best_d = require_seed_coverage("complete-dmkcn oracle-best", best_d, seeds)
    mean_d = require_seed_coverage("complete-dmkcn per-seed mean", mean_d, seeds)

    print("### 2x2 means over seeds, metric order: NMI / ARI / ACC")

    print("\n[scMUG latent]")
    print_line("(1) COMPLETE oracle-best/seed", *dict_mean_std(best_s))
    print_line("    COMPLETE per-seed all-config mean", *dict_mean_std(mean_s))
    print_line("(2) spectral direct, no C/D", *dict_mean_std(spec_s))
    print_line("    k-means direct, diagnostic", *dict_mean_std(km_s))

    print("\n[scDMKC latent]")
    print_line("(3) COMPLETE oracle-best/seed", *dict_mean_std(best_d))
    print_line("    COMPLETE per-seed all-config mean", *dict_mean_std(mean_d))
    print_line("(4) spectral direct, no C/D", *dict_mean_std(spec_d))
    print_line("    k-means direct, diagnostic", *dict_mean_std(km_d))

    print("\n### Overall COMPLETE all-config row means")
    print_line("scMUG COMPLETE all rows", all_s.mean(axis=0), all_s.std(axis=0))
    print_line("scDMKC COMPLETE all rows", all_d.mean(axis=0), all_d.std(axis=0))

    print("\n### Paired tests, Wilcoxon, delta = first - second")
    print("-- Does C/D help? Oracle-best COMPLETE vs direct spectral")
    paired(best_s, spec_s, "scMUG: COMPLETE oracle - direct", seeds)
    paired(best_d, spec_d, "scDMKC: COMPLETE oracle - direct", seeds)

    print("-- Does C/D help? Per-seed mean COMPLETE vs direct spectral")
    paired(mean_s, spec_s, "scMUG: COMPLETE mean - direct", seeds)
    paired(mean_d, spec_d, "scDMKC: COMPLETE mean - direct", seeds)

    print("-- Is scDMKC representation better? Same column comparisons")
    paired(spec_d, spec_s, "Direct: scDMKC - scMUG", seeds)
    paired(best_d, best_s, "COMPLETE oracle: scDMKC - scMUG", seeds)
    paired(mean_d, mean_s, "COMPLETE mean: scDMKC - scMUG", seeds)

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

    with open(args.outfile, "w", encoding="utf-8") as f:
        f.write("# Per-seed method results\n")
        f.write("arm\tmethod\tseed\tnmi\tari\tacc\n")

        write_method_rows(f, "scMUG", "complete_oracle_best", best_s, seeds)
        write_method_rows(f, "scMUG", "complete_per_seed_mean", mean_s, seeds)
        write_method_rows(f, "scMUG", "spectral_direct_no_CD", spec_s, seeds)
        write_method_rows(f, "scMUG", "kmeans_direct_diagnostic", km_s, seeds)

        write_method_rows(f, "scDMKC", "complete_oracle_best", best_d, seeds)
        write_method_rows(f, "scDMKC", "complete_per_seed_mean", mean_d, seeds)
        write_method_rows(f, "scDMKC", "spectral_direct_no_CD", spec_d, seeds)
        write_method_rows(f, "scDMKC", "kmeans_direct_diagnostic", km_d, seeds)

        f.write("\n# Paired comparisons\n")
        f.write(
            "comparison\tmetric\tmean_delta\tstd_delta\twins\tlosses\tties\twilcoxon_p\n"
        )

        write_paired_rows(
            f,
            "scMUG_complete_oracle_minus_direct",
            best_s,
            spec_s,
            seeds,
        )
        write_paired_rows(
            f,
            "scDMKC_complete_oracle_minus_direct",
            best_d,
            spec_d,
            seeds,
        )
        write_paired_rows(
            f,
            "scMUG_complete_mean_minus_direct",
            mean_s,
            spec_s,
            seeds,
        )
        write_paired_rows(
            f,
            "scDMKC_complete_mean_minus_direct",
            mean_d,
            spec_d,
            seeds,
        )
        write_paired_rows(
            f,
            "direct_scDMKC_minus_scMUG",
            spec_d,
            spec_s,
            seeds,
        )
        write_paired_rows(
            f,
            "complete_oracle_scDMKC_minus_scMUG",
            best_d,
            best_s,
            seeds,
        )
        write_paired_rows(
            f,
            "complete_mean_scDMKC_minus_scMUG",
            mean_d,
            mean_s,
            seeds,
        )

    print("\n### Interpretation guide")
    print(
        "  - COMPLETE oracle-best is optimistic because alpha/beta are selected using labels."
    )
    print(
        "  - COMPLETE per-seed mean is less optimistic, but averages over weak alpha/beta settings."
    )
    print(
        "  - If scDMKC direct ≈ scDMKC COMPLETE oracle, C/D adds little once scDMKC B is used."
    )
    print("  - If scMUG COMPLETE >> scMUG direct but scDMKC COMPLETE ≈ scDMKC direct,")
    print("    then scDMKC representation makes C/D less necessary.")
    print(
        "  - If direct scDMKC > direct scMUG, scDMKC improves the representation independently of C/D."
    )

    print(f"\nSummary written to {args.outfile}")


if __name__ == "__main__":
    main()
