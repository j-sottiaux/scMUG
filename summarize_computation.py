#!/usr/bin/env python3
"""Consolidate run-specific scMUG computation metrics for analysis and plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = [
    "experiment_id",
    "run_id",
    "dataset",
    "k",
    "pipeline",
    "n_cells",
    "n_genes",
    "n_hvg",
    "n_gfm",
]
NUMERIC_COLUMNS = [
    "k",
    "seed",
    "repeat",
    "gfm_index",
    "alpha",
    "beta",
    "elapsed_seconds",
    "n_cells",
    "n_genes",
    "n_hvg",
    "n_gfm",
    "gfm_gene_count",
    "peak_gpu_allocated_mb",
    "peak_gpu_reserved_mb",
    "peak_rss_mb",
]


def load_metrics(input_root: Path) -> pd.DataFrame:
    paths = sorted(input_root.glob("*/computation/*_timings.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No computation timing files found below {input_root}"
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    required = set(IDENTITY_COLUMNS + ["stage", "elapsed_seconds", "status"])
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Computation metrics are missing columns: {missing}")
    for column in NUMERIC_COLUMNS:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw["elapsed_seconds"].isna().any() or (raw["elapsed_seconds"] < 0).any():
        raise ValueError("Invalid elapsed_seconds values in computation metrics")
    return raw


def run_stage_totals(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = IDENTITY_COLUMNS + ["stage"]
    return (
        raw.groupby(group_columns, dropna=False)
        .agg(
            elapsed_seconds=("elapsed_seconds", "sum"),
            event_count=("elapsed_seconds", "size"),
            peak_gpu_allocated_mb=("peak_gpu_allocated_mb", "max"),
            peak_gpu_reserved_mb=("peak_gpu_reserved_mb", "max"),
            peak_rss_mb=("peak_rss_mb", "max"),
        )
        .reset_index()
        .sort_values(group_columns)
    )


def summarize_runs(run_stages: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "experiment_id",
        "dataset",
        "k",
        "pipeline",
        "n_cells",
        "n_genes",
        "n_hvg",
        "n_gfm",
        "stage",
    ]
    return (
        run_stages.groupby(group_columns, dropna=False)
        .agg(
            n_runs=("run_id", "nunique"),
            elapsed_median_seconds=("elapsed_seconds", "median"),
            elapsed_mean_seconds=("elapsed_seconds", "mean"),
            elapsed_std_seconds=("elapsed_seconds", "std"),
            elapsed_min_seconds=("elapsed_seconds", "min"),
            elapsed_max_seconds=("elapsed_seconds", "max"),
            peak_gpu_allocated_mb=("peak_gpu_allocated_mb", "max"),
            peak_gpu_reserved_mb=("peak_gpu_reserved_mb", "max"),
            peak_rss_mb=("peak_rss_mb", "max"),
        )
        .reset_index()
        .sort_values(group_columns)
    )


def summarize_canonical_seeds(raw: pd.DataFrame) -> pd.DataFrame:
    seeds = raw.loc[raw["stage"] == "canonical_seed_total"].copy()
    group_columns = [
        "experiment_id",
        "dataset",
        "k",
        "pipeline",
        "n_cells",
        "n_genes",
        "n_hvg",
        "n_gfm",
    ]
    return (
        seeds.groupby(group_columns, dropna=False)
        .agg(
            n_seeds=("seed", "count"),
            elapsed_median_seconds=("elapsed_seconds", "median"),
            elapsed_mean_seconds=("elapsed_seconds", "mean"),
            elapsed_std_seconds=("elapsed_seconds", "std"),
            elapsed_min_seconds=("elapsed_seconds", "min"),
            elapsed_max_seconds=("elapsed_seconds", "max"),
            peak_gpu_allocated_mb=("peak_gpu_allocated_mb", "max"),
            peak_gpu_reserved_mb=("peak_gpu_reserved_mb", "max"),
            peak_rss_mb=("peak_rss_mb", "max"),
        )
        .reset_index()
        .sort_values(group_columns)
    )


def compare_pipeline_costs(canonical_seeds: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "experiment_id",
        "dataset",
        "k",
        "n_cells",
        "n_genes",
        "n_hvg",
        "n_gfm",
    ]
    pivot = canonical_seeds.pivot_table(
        index=index_columns,
        columns="pipeline",
        values="elapsed_median_seconds",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    if "scMUG" not in pivot.columns or "scMUG-DMKCN" not in pivot.columns:
        pivot["plugin_minus_scmug_seconds"] = pd.NA
        pivot["plugin_over_scmug_ratio"] = pd.NA
        return pivot
    pivot["plugin_minus_scmug_seconds"] = (
        pivot["scMUG-DMKCN"] - pivot["scMUG"]
    )
    pivot["plugin_over_scmug_ratio"] = pivot["scMUG-DMKCN"] / pivot["scMUG"]
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()

    raw = load_metrics(args.input_root)
    if args.experiment_id is not None:
        raw = raw.loc[raw["experiment_id"] == args.experiment_id].copy()
        if raw.empty:
            raise ValueError(
                f"No computation rows for experiment_id={args.experiment_id}"
            )

    stages = run_stage_totals(raw)
    summary = summarize_runs(stages)
    canonical_seeds = summarize_canonical_seeds(raw)
    pipeline_costs = compare_pipeline_costs(canonical_seeds)
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.outdir / "computation_raw.csv", index=False)
    stages.to_csv(args.outdir / "computation_run_stage_totals.csv", index=False)
    summary.to_csv(args.outdir / "computation_summary.csv", index=False)

    canonical = summary.loc[summary["stage"] == "canonical_pipeline_total"]
    canonical.to_csv(
        args.outdir / "computation_pipeline_for_plot.csv", index=False
    )
    canonical_seeds.to_csv(
        args.outdir / "computation_seed_pipeline_for_plot.csv", index=False
    )
    pipeline_costs.to_csv(
        args.outdir / "computation_pipeline_cost_comparison.csv", index=False
    )
    print(f"Loaded {len(raw)} timing rows from {args.input_root}")
    print(f"Canonical pipeline rows for plots: {len(canonical)}")
    print(f"Wrote computation summaries to {args.outdir}")


if __name__ == "__main__":
    main()
