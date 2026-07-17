"""Compare scMUG and scMUG-DMKCN after k and alpha/beta selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from summarize_k_sweep import load_dataset_config
from xp_statistics import fdr_bh


METRICS = ("NMI", "ARI", "ACC")
PIPELINES = ("scMUG", "scMUG-DMKCN")


def validate_pair_coverage(table: pd.DataFrame, datasets: list[str], label: str) -> None:
    expected = {(dataset, model) for dataset in datasets for model in PIPELINES}
    observed = set(zip(table["dataset"], table["model"]))
    if observed != expected or len(table) != len(expected):
        raise ValueError(
            f"{label} must contain one row per dataset and pipeline. Missing: "
            f"{sorted(expected - observed)}; unexpected: {sorted(observed - expected)}"
        )


def model_comparison(table: pd.DataFrame, comparison: str) -> pd.DataFrame:
    left = table.loc[table["model"] == "scMUG"].copy()
    right = table.loc[table["model"] == "scMUG-DMKCN"].copy()
    columns = ["dataset", "k", "alpha", "beta"] + [
        f"{metric}_median" for metric in METRICS
    ]
    merged = left[columns].merge(
        right[columns], on="dataset", suffixes=("_scMUG", "_scMUG_DMKCN")
    )
    merged.insert(1, "comparison", comparison)
    for metric in METRICS:
        merged[f"{metric}_median_delta_plugin_minus_scMUG"] = (
            merged[f"{metric}_median_scMUG_DMKCN"]
            - merged[f"{metric}_median_scMUG"]
        )
    return merged.sort_values("dataset").reset_index(drop=True)


def select_seed_configuration(
    seed_scores: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    frames = []
    for row in selected.itertuples(index=False):
        mask = (
            (seed_scores["dataset"] == row.dataset)
            & (seed_scores["model"] == row.model)
            & (seed_scores["k"] == row.k)
            & np.isclose(seed_scores["alpha"], row.alpha)
            & np.isclose(seed_scores["beta"], row.beta)
        )
        frame = seed_scores.loc[mask].copy()
        if frame.empty:
            raise ValueError(
                f"No seed scores for {row.dataset}, {row.model}, k={row.k}, "
                f"alpha={row.alpha}, beta={row.beta}."
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def paired_seed_comparison(seed_scores: pd.DataFrame, comparison: str) -> pd.DataFrame:
    rows = []
    for dataset, frame in seed_scores.groupby("dataset"):
        ae = frame.loc[frame["model"] == "scMUG"].set_index("seed")
        plugin = frame.loc[frame["model"] == "scMUG-DMKCN"].set_index("seed")
        common = sorted(set(ae.index).intersection(plugin.index))
        if len(common) < 2:
            raise ValueError(f"{dataset}: fewer than two paired seeds.")
        if len(ae) != len(common) or len(plugin) != len(common):
            raise ValueError(f"{dataset}: unpaired or duplicated seed rows.")
        for metric in METRICS:
            diff = (
                plugin.loc[common, metric].to_numpy(dtype=float)
                - ae.loc[common, metric].to_numpy(dtype=float)
            )
            if np.allclose(diff, 0):
                statistic, p_value = np.nan, 1.0
            else:
                statistic, p_value = wilcoxon(diff)
            rows.append(
                {
                    "comparison": comparison,
                    "dataset": dataset,
                    "metric": metric,
                    "direction": "scMUG-DMKCN_minus_scMUG",
                    "n_paired_seeds": len(common),
                    "mean_difference": float(np.mean(diff)),
                    "median_difference": float(np.median(diff)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    out = pd.DataFrame(rows)
    out["p_value_fdr_bh"] = out.groupby("metric")["p_value"].transform(
        lambda values: fdr_bh(values.to_numpy(dtype=float))
    )
    return out.sort_values(["metric", "dataset"]).reset_index(drop=True)


def canonical_recomputation_consistency(
    alpha_summary: pd.DataFrame,
    k_scores: pd.DataFrame,
    best_k: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    recomputed_rows = []
    for row in best_k.itertuples(index=False):
        alpha = float(config[row.dataset]["full_alpha"])
        beta = float(config[row.dataset]["full_beta"])
        mask = (
            (alpha_summary["dataset"] == row.dataset)
            & (alpha_summary["model"] == row.model)
            & (alpha_summary["k"] == row.k)
            & np.isclose(alpha_summary["alpha"], alpha)
            & np.isclose(alpha_summary["beta"], beta)
        )
        selected = alpha_summary.loc[mask]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one canonical recomputed row for {row.dataset}, "
                f"{row.model}, k={row.k}; found {len(selected)}."
            )
        recomputed_rows.append(selected.iloc[0])
    recomputed = pd.DataFrame(recomputed_rows)
    direct = k_scores.merge(
        best_k[["dataset", "model", "k"]], on=["dataset", "model", "k"]
    )
    columns = ["dataset", "model", "k", "alpha", "beta"] + [
        f"{metric}_median" for metric in METRICS
    ]
    merged = direct[columns].merge(
        recomputed[columns],
        on=["dataset", "model", "k", "alpha", "beta"],
        suffixes=("_direct", "_recomputed"),
    )
    for metric in METRICS:
        merged[f"{metric}_recomputed_minus_direct"] = (
            merged[f"{metric}_median_recomputed"]
            - merged[f"{metric}_median_direct"]
        )
    return merged.sort_values(["dataset", "model"]).reset_index(drop=True)


def select_reference_k(
    table: pd.DataFrame,
    config: dict,
    config_key: str,
    label: str,
    *,
    require_unique_pair: bool = True,
) -> pd.DataFrame:
    reference = {
        dataset: int(params[config_key]) for dataset, params in config.items()
    }
    selected = table.loc[
        table.apply(
            lambda row: int(row["k"]) == reference[row["dataset"]], axis=1
        )
    ].copy()
    expected = {
        (dataset, model) for dataset in config for model in PIPELINES
    }
    observed = set(zip(selected["dataset"], selected["model"]))
    if observed != expected:
        raise ValueError(
            f"{label} has invalid dataset/pipeline coverage. Missing: "
            f"{sorted(expected - observed)}; unexpected: {sorted(observed - expected)}"
        )
    if require_unique_pair:
        validate_pair_coverage(selected, list(config), label)
    return selected


def validate_canonical_consistency(
    consistency: pd.DataFrame, max_abs_nmi_delta: float
) -> None:
    if max_abs_nmi_delta < 0:
        raise ValueError("Canonical NMI tolerance must be non-negative.")
    delta_column = "NMI_recomputed_minus_direct"
    deltas = pd.to_numeric(consistency[delta_column], errors="raise").abs()
    if not np.isfinite(deltas).all():
        raise ValueError("Canonical direct/recomputed NMI deltas must be finite.")
    invalid = consistency.loc[deltas > max_abs_nmi_delta]
    if not invalid.empty:
        details = invalid[["dataset", "model", "k", delta_column]].to_dict("records")
        raise ValueError(
            "Canonical direct/recomputed NMI consistency threshold exceeded "
            f"(tolerance={max_abs_nmi_delta}): {details}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--best-k",
        type=Path,
        default=Path("publication/k_sweep/ksweep_best_k_by_nmi.csv"),
    )
    parser.add_argument(
        "--k-scores",
        type=Path,
        default=Path("publication/k_sweep/ksweep_full_pipeline_scores.csv"),
    )
    parser.add_argument(
        "--k-seed-scores",
        type=Path,
        default=Path("publication/k_sweep/ksweep_seed_scores.csv"),
    )
    parser.add_argument(
        "--alpha-summary",
        type=Path,
        default=Path("publication/alpha_beta/alpha_beta_summary.csv"),
    )
    parser.add_argument(
        "--alpha-seed-scores",
        type=Path,
        default=Path("publication/alpha_beta/alpha_beta_seed_means.csv"),
    )
    parser.add_argument(
        "--best-alpha-beta",
        type=Path,
        default=Path("publication/alpha_beta/alpha_beta_best_by_nmi.csv"),
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("publication/final_comparison")
    )
    parser.add_argument(
        "--max-canonical-nmi-delta",
        type=float,
        default=0.01,
        help="Maximum absolute median-NMI delta between direct and recomputed C/D.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.config)
    datasets = list(config)
    best_k = pd.read_csv(args.best_k)
    k_scores = pd.read_csv(args.k_scores)
    k_seed_scores = pd.read_csv(args.k_seed_scores)
    alpha_summary = pd.read_csv(args.alpha_summary)
    alpha_seed_scores = pd.read_csv(args.alpha_seed_scores)
    best_pairs = pd.read_csv(args.best_alpha_beta)

    validate_pair_coverage(best_k, datasets, "Best-k table")
    validate_pair_coverage(best_pairs, datasets, "Best-alpha/beta table")

    optimized = model_comparison(best_pairs, "optimized_per_pipeline")
    optimized_seed_scores = select_seed_configuration(alpha_seed_scores, best_pairs)
    optimized_paired = paired_seed_comparison(
        optimized_seed_scores, "optimized_per_pipeline"
    )

    published_reference = select_reference_k(
        k_scores,
        config,
        "published_best_k",
        "Published-best-k controlled table",
    )
    published_reference_comparison = model_comparison(
        published_reference, "published_best_k_published_alpha_beta"
    )
    published_reference_seed = select_reference_k(
        k_seed_scores,
        config,
        "published_best_k",
        "Published-best-k seed table",
        require_unique_pair=False,
    )
    published_reference_paired = paired_seed_comparison(
        published_reference_seed, "published_best_k_published_alpha_beta"
    )

    ground_truth = select_reference_k(
        k_scores,
        config,
        "ground_truth_k",
        "Ground-truth-k controlled table",
    )
    ground_truth_comparison = model_comparison(
        ground_truth, "ground_truth_k_published_alpha_beta"
    )
    ground_truth_seed = select_reference_k(
        k_seed_scores,
        config,
        "ground_truth_k",
        "Ground-truth-k seed table",
        require_unique_pair=False,
    )
    ground_truth_paired = paired_seed_comparison(
        ground_truth_seed, "ground_truth_k_published_alpha_beta"
    )

    consistency = canonical_recomputation_consistency(
        alpha_summary, k_scores, best_k, config
    )
    validate_canonical_consistency(consistency, args.max_canonical_nmi_delta)

    args.outdir.mkdir(parents=True, exist_ok=True)
    optimized.to_csv(args.outdir / "optimized_model_comparison.csv", index=False)
    optimized_paired.to_csv(
        args.outdir / "optimized_paired_seed_tests.csv", index=False
    )
    published_reference_comparison.to_csv(
        args.outdir / "controlled_published_best_k_comparison.csv", index=False
    )
    published_reference_paired.to_csv(
        args.outdir / "controlled_published_best_k_paired_seed_tests.csv", index=False
    )
    ground_truth_comparison.to_csv(
        args.outdir / "controlled_ground_truth_k_comparison.csv", index=False
    )
    ground_truth_paired.to_csv(
        args.outdir / "controlled_ground_truth_k_paired_seed_tests.csv", index=False
    )
    consistency.to_csv(
        args.outdir / "canonical_recomputation_consistency.csv", index=False
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "datasets": datasets,
        "selection_metric": "NMI_median",
        "comparisons": [
            "optimized_per_pipeline",
            "published_best_k_published_alpha_beta",
            "ground_truth_k_published_alpha_beta",
            "canonical_direct_vs_recomputed_from_latents",
        ],
        "max_canonical_nmi_delta": args.max_canonical_nmi_delta,
        "paired_test": "two-sided Wilcoxon signed-rank across matched seeds",
        "fdr": "Benjamini-Hochberg separately across datasets for each metric",
        "inputs": {
            key: str(value)
            for key, value in vars(args).items()
            if key != "outdir"
        },
    }
    with (args.outdir / "final_comparison_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(optimized.to_string(index=False))
    print(f"\nWrote final comparison tables to {args.outdir}")


if __name__ == "__main__":
    main()
