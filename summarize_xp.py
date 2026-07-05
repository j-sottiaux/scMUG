import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def fdr_bh(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    valid = ~np.isnan(p)
    pv = p[valid]

    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted
    q[valid] = q_valid

    return q


def read_full_txt(
    path, model_name, canonical_alpha, canonical_beta, condition_name="full"
):
    rows = []

    pattern = re.compile(
        r"dbname:(?P<dataset>\S+)\s+"
        r"round:(?P<seed>\S+)\s+"
        r"alpha:(?P<alpha>\S+)\s+"
        r"beta:(?P<beta>\S+)\s+"
        r"acc:\s*(?P<acc>[0-9.]+)\s+"
        r"ari:\s*(?P<ari>[0-9.]+)\s+"
        r"nmi:\s*(?P<nmi>[0-9.]+)"
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match is None:
                continue

            d = match.groupdict()

            rows.append(
                {
                    "dataset": d["dataset"],
                    "model": model_name,
                    "condition": condition_name,
                    "seed": int(float(d["seed"])),
                    "alpha": float(d["alpha"]),
                    "beta": float(d["beta"]),
                    "acc": float(d["acc"]),
                    "ari": float(d["ari"]),
                    "nmi": float(d["nmi"]),
                    "source": "full_run",
                }
            )

    if not rows:
        raise ValueError(f"No valid rows parsed from {path}")

    df = pd.DataFrame(rows)

    mask = (df["alpha"] == canonical_alpha) & (df["beta"] == canonical_beta)

    if not mask.any():
        available_pairs = sorted(set(zip(df["alpha"], df["beta"])))
        raise ValueError(
            f"{path}: no rows with alpha={canonical_alpha}, beta={canonical_beta}. "
            f"Available pairs: {available_pairs}"
        )

    return df[mask].reset_index(drop=True)


def read_ablation_tsv(path):
    df = pd.read_csv(path, sep="\t")

    required = {"arm", "seed", "method", "acc", "ari", "nmi"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    out = pd.DataFrame(
        {
            "dataset": "unknown",
            "model": df["arm"].replace(
                {
                    "autoencoder": "scMUG",
                    "dmkcn": "scMUG-DMKCN",
                }
            ),
            "condition": df["method"].replace(
                {
                    "no_C": "without_C",
                    "no_D": "without_D",
                    "no_CD_direct_spectral": "without_CD_direct_spectral",
                }
            ),
            "seed": df["seed"].astype(int),
            "alpha": df["alpha"] if "alpha" in df.columns else pd.NA,
            "beta": df["beta"] if "beta" in df.columns else pd.NA,
            "acc": df["acc"].astype(float),
            "ari": df["ari"].astype(float),
            "nmi": df["nmi"].astype(float),
            "source": "ablation",
        }
    )

    return out


def summarize(df):
    summary = (
        df.groupby(
            ["dataset", "model", "condition", "alpha", "beta"],
            dropna=False,
        )
        .agg(
            n_runs=("ari", "count"),
            acc_mean=("acc", "mean"),
            acc_best=("acc", "max"),
            ari_mean=("ari", "mean"),
            ari_best=("ari", "max"),
            nmi_mean=("nmi", "mean"),
            nmi_best=("nmi", "max"),
        )
        .reset_index()
    )

    order_model = {
        "scMUG": 0,
        "scMUG-DMKCN": 1,
    }

    order_condition = {
        "full": 0,
        "without_C": 1,
        "without_D": 2,
        "without_CD_direct_spectral": 3,
    }

    summary["_model_order"] = summary["model"].map(order_model)
    summary["_condition_order"] = summary["condition"].map(order_condition)

    summary = (
        summary.sort_values(
            ["dataset", "_model_order", "_condition_order", "alpha", "beta"]
        )
        .drop(columns=["_model_order", "_condition_order"])
        .reset_index(drop=True)
    )

    return summary


def seed_means(df):
    return (
        df.groupby(
            ["dataset", "model", "condition", "seed"],
            dropna=False,
        )[["nmi", "ari", "acc"]]
        .mean()
        .reset_index()
    )


def paired_wilcoxon(
    group_a, group_b, label_a, label_b, family, metrics=("nmi", "ari", "acc")
):
    common_seeds = sorted(set(group_a["seed"]) & set(group_b["seed"]))

    if len(common_seeds) < 2:
        return []

    group_a = group_a.set_index("seed").loc[common_seeds]
    group_b = group_b.set_index("seed").loc[common_seeds]

    rows = []

    for metric in metrics:
        diff = group_a[metric].to_numpy(dtype=float) - group_b[metric].to_numpy(
            dtype=float
        )

        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0
        else:
            try:
                stat, p = wilcoxon(diff)
            except ValueError:
                stat, p = np.nan, np.nan

        rows.append(
            {
                "family": family,
                "group_a": label_a,
                "group_b": label_b,
                "direction": "group_a_minus_group_b",
                "metric": metric,
                "n_pairs": len(common_seeds),
                "mean_diff": float(np.mean(diff)),
                "median_diff": float(np.median(diff)),
                "wins_group_a": int((diff > 0).sum()),
                "wins_group_b": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "wilcoxon_stat": stat,
                "p_value": p,
            }
        )

    return rows


def build_significance_table(all_rows):
    sm = seed_means(all_rows)

    if sm.empty:
        return pd.DataFrame()

    dataset = sm["dataset"].iloc[0]

    ablated_conditions = [
        "without_C",
        "without_D",
        "without_CD_direct_spectral",
    ]

    models = ["scMUG", "scMUG-DMKCN"]

    def get(model, condition):
        return sm[(sm["model"] == model) & (sm["condition"] == condition)]

    records = []

    for model in models:
        full = get(model, "full")

        for condition in ablated_conditions:
            ablated = get(model, condition)

            records.extend(
                paired_wilcoxon(
                    full,
                    ablated,
                    label_a=f"{model}:full",
                    label_b=f"{model}:{condition}",
                    family="ablation_impact",
                )
            )

    for condition in ["full"] + ablated_conditions:
        scmug = get("scMUG", condition)
        dmkcn = get("scMUG-DMKCN", condition)

        records.extend(
            paired_wilcoxon(
                scmug,
                dmkcn,
                label_a=f"scMUG:{condition}",
                label_b=f"scMUG-DMKCN:{condition}",
                family="model_comparison",
            )
        )

    significance = pd.DataFrame(records)

    if significance.empty:
        return significance

    significance.insert(0, "dataset", dataset)

    # FDR-correct within each hypothesis family separately: mixing "does the
    # ablation hurt" tests with "which model wins" tests in one pool would let
    # one family's signal dilute or inflate the other's corrected p-values.
    significance["p_value_fdr_bh"] = significance.groupby("family")["p_value"].transform(
        lambda p: fdr_bh(p.to_numpy())
    )

    return significance


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--full-autoencoder", required=True)
    parser.add_argument("--full-dmkcn", required=True)
    parser.add_argument("--ablation", required=True)

    parser.add_argument("--outfile", default=None)
    parser.add_argument("--significance-outfile", default=None)

    parser.add_argument(
        "--full-alpha",
        default=1.0,
        type=float,
        help="Canonical alpha used for the full pipeline.",
    )

    parser.add_argument(
        "--full-beta",
        default=1.0,
        type=float,
        help="Canonical beta used for the full pipeline.",
    )

    args = parser.parse_args()

    full_auto = read_full_txt(
        args.full_autoencoder,
        model_name="scMUG",
        canonical_alpha=args.full_alpha,
        canonical_beta=args.full_beta,
        condition_name="full",
    )

    full_dmkcn = read_full_txt(
        args.full_dmkcn,
        model_name="scMUG-DMKCN",
        canonical_alpha=args.full_alpha,
        canonical_beta=args.full_beta,
        condition_name="full",
    )

    ablation = read_ablation_tsv(args.ablation)

    full_auto["dataset"] = args.dataset
    full_dmkcn["dataset"] = args.dataset
    ablation["dataset"] = args.dataset

    all_rows = pd.concat(
        [full_auto, full_dmkcn, ablation],
        ignore_index=True,
    )

    summary = summarize(all_rows)

    if args.outfile is None:
        args.outfile = f"./outputs/summary_comparison_{args.dataset}.csv"

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

    summary.to_csv(args.outfile, index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote: {args.outfile}")

    significance = build_significance_table(all_rows)

    if args.significance_outfile is None:
        args.significance_outfile = f"./outputs/significance_{args.dataset}.csv"

    os.makedirs(os.path.dirname(args.significance_outfile) or ".", exist_ok=True)

    significance.to_csv(args.significance_outfile, index=False)

    if not significance.empty:
        print("\n" + significance.to_string(index=False))
    else:
        print("\nNo significance results produced.")

    print(f"\nWrote: {args.significance_outfile}")


if __name__ == "__main__":
    main()
