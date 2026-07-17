"""Validate and summarize the canonical-alpha/beta scMUG k sweep."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PIPELINES = ("scMUG", "scMUG-DMKCN")
REQUIRED_COLUMNS = {
    "experiment_id",
    "run_id",
    "dataset",
    "k",
    "model",
    "condition",
    "alpha",
    "beta",
    "NMI_median",
    "ARI_median",
    "ACC_median",
    "n_seeds",
}
SEED_REQUIRED_COLUMNS = {
    "experiment_id",
    "run_id",
    "dataset",
    "k",
    "model",
    "condition",
    "seed",
    "alpha",
    "beta",
    "NMI",
    "ARI",
    "ACC",
}


def load_dataset_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config:
        raise ValueError("Dataset configuration must be a non-empty mapping.")
    datasets = list(config)
    if datasets != sorted(datasets, key=str.casefold):
        raise ValueError("Dataset configuration must be alphabetically ordered.")
    return config


def discover_scores(input_root: Path, experiment_id: str) -> pd.DataFrame:
    paths = sorted(input_root.glob("*/results/*_canonical_condition_summary.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "experiment_id" not in frame.columns or not (
            frame["experiment_id"] == experiment_id
        ).any():
            continue
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.loc[
            (frame["experiment_id"] == experiment_id)
            & (frame["condition"] == "full")
            & (frame["model"].isin(PIPELINES))
        ].copy()
        if frame.empty:
            continue
        frame["source_summary_file"] = str(path)
        frame["source_mtime_ns"] = path.stat().st_mtime_ns
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No canonical full-pipeline summaries found for experiment_id="
            f"{experiment_id!r} below {input_root}."
        )
    return pd.concat(frames, ignore_index=True)


def discover_seed_scores(input_root: Path, experiment_id: str) -> pd.DataFrame:
    paths = sorted(input_root.glob("*/results/*_canonical_seed_means.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "experiment_id" not in frame.columns or not (
            frame["experiment_id"] == experiment_id
        ).any():
            continue
        missing = SEED_REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.loc[
            (frame["experiment_id"] == experiment_id)
            & (frame["condition"] == "full")
            & (frame["model"].isin(PIPELINES))
        ].copy()
        if frame.empty:
            continue
        frame["source_seed_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No canonical seed summaries found for experiment_id={experiment_id!r} "
            f"below {input_root}."
        )
    return pd.concat(frames, ignore_index=True)


def select_and_validate_seed_scores(
    seed_scores: pd.DataFrame,
    selected_runs: pd.DataFrame,
    expected_seeds: int,
) -> pd.DataFrame:
    run_keys = selected_runs[["dataset", "k", "model", "run_id"]].drop_duplicates()
    selected = seed_scores.merge(
        run_keys,
        on=["dataset", "k", "model", "run_id"],
        how="inner",
    )
    key = ["dataset", "k", "model", "seed"]
    if selected.duplicated(key).any():
        raise ValueError("Duplicate per-seed k-sweep scores remain after run selection.")
    counts = selected.groupby(["dataset", "k", "model"])["seed"].nunique()
    invalid = counts.loc[counts != expected_seeds]
    if not invalid.empty:
        raise ValueError(
            f"Expected {expected_seeds} seed scores per dataset/k/model: "
            f"{invalid.to_dict()}"
        )
    if len(counts) != len(selected_runs):
        raise ValueError(
            "Per-seed score coverage does not match the selected condition summaries."
        )
    return selected.sort_values(key).reset_index(drop=True)


def validate_and_select_runs(
    scores: pd.DataFrame,
    config: dict,
    expected_seeds: int,
    duplicate_policy: str,
) -> pd.DataFrame:
    scores = scores.copy()
    for column in [
        "k",
        "alpha",
        "beta",
        "NMI_median",
        "ARI_median",
        "ACC_median",
        "n_seeds",
    ]:
        scores[column] = pd.to_numeric(scores[column], errors="raise")

    unknown = sorted(set(scores["dataset"]).difference(config))
    if unknown:
        raise ValueError(f"Results contain datasets absent from the config: {unknown}")

    for dataset, frame in scores.groupby("dataset"):
        expected_alpha = float(config[dataset]["full_alpha"])
        expected_beta = float(config[dataset]["full_beta"])
        valid = np.isclose(frame["alpha"], expected_alpha) & np.isclose(
            frame["beta"], expected_beta
        )
        if not valid.all():
            observed = sorted(set(zip(frame["alpha"], frame["beta"])))
            raise ValueError(
                f"{dataset}: expected published pair "
                f"({expected_alpha}, {expected_beta}), observed {observed}."
            )

    invalid_seed_counts = scores.loc[scores["n_seeds"] != expected_seeds]
    if not invalid_seed_counts.empty:
        details = invalid_seed_counts[
            ["dataset", "k", "model", "run_id", "n_seeds"]
        ].to_dict("records")
        raise ValueError(
            f"Every result must consolidate {expected_seeds} seeds: {details}"
        )

    key = ["dataset", "k", "model"]
    duplicates = scores.duplicated(key, keep=False)
    if duplicates.any():
        details = scores.loc[duplicates, key + ["run_id", "source_summary_file"]]
        if duplicate_policy == "error":
            raise ValueError(
                "Duplicate completed runs found. Resolve them or rerun with "
                "--duplicate-policy latest:\n"
                + details.to_string(index=False)
            )
        scores = (
            scores.sort_values(["source_mtime_ns", "run_id"])
            .drop_duplicates(key, keep="last")
            .copy()
        )

    return scores.sort_values(key).reset_index(drop=True)


def coverage_table(
    scores: pd.DataFrame,
    config: dict,
    k_min: int,
    k_max: int,
) -> pd.DataFrame:
    expected_k = set(range(k_min, k_max + 1))
    rows = []
    for dataset in config:
        for model in PIPELINES:
            observed = set(
                scores.loc[
                    (scores["dataset"] == dataset) & (scores["model"] == model),
                    "k",
                ].astype(int)
            )
            missing = sorted(expected_k.difference(observed))
            unexpected = sorted(observed.difference(expected_k))
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "k_min": k_min,
                    "k_max": k_max,
                    "n_expected": len(expected_k),
                    "n_observed": len(observed.intersection(expected_k)),
                    "observed_k": json.dumps(sorted(observed)),
                    "missing_k": json.dumps(missing),
                    "unexpected_k": json.dumps(unexpected),
                    "complete": not missing and not unexpected,
                }
            )
    return pd.DataFrame(rows)


def select_best_k(scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    ranked = scores.sort_values(
        ["dataset", "model", "NMI_median", "ARI_median", "ACC_median", "k"],
        ascending=[True, True, False, False, False, True],
    ).copy()
    ranked["nmi_rank"] = ranked.groupby(["dataset", "model"]).cumcount() + 1
    best = ranked.loc[ranked["nmi_rank"] == 1].copy()
    best["published_best_k"] = best["dataset"].map(
        {dataset: int(params["published_best_k"]) for dataset, params in config.items()}
    )
    best["ground_truth_k"] = best["dataset"].map(
        {dataset: int(params["ground_truth_k"]) for dataset, params in config.items()}
    )
    best["delta_from_published_best_k"] = best["k"] - best["published_best_k"]
    best["matches_published_best_k"] = best["delta_from_published_best_k"] == 0
    best["delta_from_ground_truth"] = best["k"] - best["ground_truth_k"]
    best["matches_ground_truth"] = best["delta_from_ground_truth"] == 0
    best["selection_metric"] = "NMI_median"
    best["tie_break_rule"] = "ARI_median,ACC_median,lower_k"
    return best.sort_values(["dataset", "model"]).reset_index(drop=True)


def attach_artifact_paths(best: pd.DataFrame, input_root: Path) -> pd.DataFrame:
    """Reference the latent files required by the subsequent alpha/beta grid."""
    best = best.copy()
    tags = best["model"].map({"scMUG": "ae", "scMUG-DMKCN": "dmkcn"})
    stems = [
        f"{experiment_id}_k{int(k):02d}_{run_id}"
        for experiment_id, k, run_id in zip(
            best["experiment_id"], best["k"], best["run_id"]
        )
    ]
    paths = [
        input_root / dataset / "artifacts" / f"{stem}_{tag}_latents.joblib"
        for dataset, stem, tag in zip(best["dataset"], stems, tags)
    ]
    best["artifact_stem"] = stems
    best["latent_artifact"] = [str(path) for path in paths]
    best["latent_artifact_exists"] = [path.is_file() for path in paths]
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--experiment-id", default="ksweep")
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument(
        "--duplicate-policy",
        choices=["error", "latest"],
        default="error",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a provisional summary even when some dataset/model/k runs are absent.",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("publication/k_sweep")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_min < 1 or args.k_max < args.k_min:
        raise ValueError(f"Invalid k range: {args.k_min}-{args.k_max}")

    config = load_dataset_config(args.config)
    discovered = discover_scores(args.input_root, args.experiment_id)
    scores = validate_and_select_runs(
        discovered,
        config,
        expected_seeds=args.expected_seeds,
        duplicate_policy=args.duplicate_policy,
    )
    seed_scores = select_and_validate_seed_scores(
        discover_seed_scores(args.input_root, args.experiment_id),
        scores,
        args.expected_seeds,
    )
    coverage = coverage_table(scores, config, args.k_min, args.k_max)
    incomplete = coverage.loc[~coverage["complete"]]
    if not incomplete.empty and not args.allow_incomplete:
        raise ValueError(
            "The k sweep is incomplete. Use --allow-incomplete only for provisional "
            "inspection:\n"
            + incomplete[["dataset", "model", "missing_k", "unexpected_k"]].to_string(
                index=False
            )
        )

    best = attach_artifact_paths(select_best_k(scores, config), args.input_root)
    args.outdir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.outdir / "ksweep_full_pipeline_scores.csv", index=False)
    seed_scores.to_csv(args.outdir / "ksweep_seed_scores.csv", index=False)
    coverage.to_csv(args.outdir / "ksweep_coverage.csv", index=False)
    best.to_csv(args.outdir / "ksweep_best_k_by_nmi.csv", index=False)
    metrics_for_plot = scores[
        [
            "dataset",
            "model",
            "k",
            "NMI_median",
            "ARI_median",
            "ACC_median",
            "n_seeds",
        ]
    ].copy()
    metrics_for_plot["published_best_k"] = metrics_for_plot["dataset"].map(
        {dataset: int(params["published_best_k"]) for dataset, params in config.items()}
    )
    metrics_for_plot["ground_truth_k"] = metrics_for_plot["dataset"].map(
        {dataset: int(params["ground_truth_k"]) for dataset, params in config.items()}
    )
    metrics_for_plot.to_csv(
        args.outdir / "ksweep_metrics_for_plot.csv", index=False
    )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment_id": args.experiment_id,
        "input_root": str(args.input_root),
        "config": str(args.config),
        "datasets": list(config),
        "pipelines": list(PIPELINES),
        "k_range": [args.k_min, args.k_max],
        "expected_seeds": args.expected_seeds,
        "duplicate_policy": args.duplicate_policy,
        "allow_incomplete": args.allow_incomplete,
        "selection_metric": "NMI_median",
        "tie_break_rule": ["ARI_median", "ACC_median", "lower_k"],
        "n_score_rows": int(len(scores)),
        "n_seed_score_rows": int(len(seed_scores)),
        "coverage_complete": bool(coverage["complete"].all()),
    }
    with (args.outdir / "ksweep_summary_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(coverage.to_string(index=False))
    print("\nSelected k values:")
    print(
        best[
            [
                "dataset",
                "model",
                "k",
                "NMI_median",
                "ARI_median",
                "ACC_median",
                "published_best_k",
                "delta_from_published_best_k",
                "ground_truth_k",
                "delta_from_ground_truth",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote k-sweep summaries to {args.outdir}")


if __name__ == "__main__":
    main()
