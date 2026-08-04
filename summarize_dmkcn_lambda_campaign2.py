#!/usr/bin/env python3
"""Validate and consolidate Darmanis/Li/Manno scMUG-DMKCN campaign-2 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import yaml

from dmkcn.lambda_config import LambdaConfiguration, load_lambda_configuration
from summarize_alpha_beta import DEFAULT_GRID
from xp_statistics import fdr_bh


DATASETS = ("darmanis", "li", "manno")
MODEL = "scMUG-DMKCN"
ARM = "dmkcn"
METRICS = ("NMI", "ARI", "ACC")
SEEDS = (1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000)
REPEAT = 3
K_MIN = 5
K_MAX = 20
LAMBDA_COLUMNS = ("lambda_config_id", "lambda1", "lambda2", "lambda3")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Non-parsable JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must contain an object: {path}")
    return payload


def read_csv(path: Path, *, sep: str = ",") -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required result file is absent or empty: {path}")
    try:
        return pd.read_csv(path, sep=sep)
    except Exception as exc:
        raise ValueError(f"Non-parsable tabular result: {path}") from exc


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_data_config(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset configuration must be a mapping: {path}")
    missing = sorted(set(DATASETS).difference(config))
    if missing:
        raise ValueError(f"Campaign datasets absent from {path}: {missing}")
    return config


def validate_lambda_scope(config: LambdaConfiguration) -> None:
    if tuple(config.datasets) != DATASETS or config.fallback != "error":
        raise ValueError(
            "Campaign-2 lambda configuration must contain exactly Darmanis, Li "
            "and Manno with fallback set to error."
        )


def manifest_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*/artifacts/*_campaign_manifest.json"))


def load_manifests(
    root: Path,
    *,
    phase: str,
    lambda_config: LambdaConfiguration,
) -> list[tuple[Path, dict[str, Any]]]:
    paths = manifest_paths(root)
    if not paths:
        raise FileNotFoundError(f"No campaign manifests found below {root}")
    loaded = []
    bad_status = []
    for path in paths:
        manifest = read_json(path)
        if manifest.get("phase") != phase:
            raise ValueError(
                f"{path}: expected phase={phase!r}, observed {manifest.get('phase')!r}."
            )
        if manifest.get("status") != "completed":
            bad_status.append((str(path), manifest.get("status")))
            continue
        dataset = manifest.get("dataset")
        if dataset not in DATASETS:
            raise ValueError(f"{path}: unexpected campaign dataset {dataset!r}.")
        expected = lambda_config.resolve(dataset)
        lambdas = manifest.get("lambdas")
        if not isinstance(lambdas, dict):
            raise ValueError(f"{path}: missing lambdas mapping.")
        for key in LAMBDA_COLUMNS:
            expected_value = getattr(expected, key)
            observed_value = lambdas.get(key)
            if key == "lambda_config_id":
                valid = observed_value == expected_value
            else:
                valid = np.isclose(float(observed_value), float(expected_value))
            if not valid:
                raise ValueError(
                    f"{path}: {key}={observed_value!r} does not match "
                    f"{expected_value!r} from {lambda_config.source_file}."
                )
        loaded.append((path, manifest))
    if bad_status:
        raise ValueError(
            "Incomplete or failed campaign manifests must be archived or resolved "
            f"before consolidation: {bad_status}"
        )
    return loaded


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def validate_expected_seeds(
    frame: pd.DataFrame,
    keys: list[str],
    source: Path,
    *,
    rows_per_seed: int = 1,
) -> None:
    expected = set(SEEDS)
    for identity, group in frame.groupby(keys, dropna=False):
        observed = set(pd.to_numeric(group["seed"], errors="raise").astype(int))
        counts = group.groupby("seed").size()
        if (
            observed != expected
            or len(group) != len(expected) * rows_per_seed
            or not (counts == rows_per_seed).all()
        ):
            raise ValueError(
                f"{source}: seed coverage mismatch for {identity}; "
                f"missing={sorted(expected - observed)}, "
                f"unexpected={sorted(observed - expected)}, rows={len(group)}, "
                f"rows_per_seed={counts.to_dict()}."
            )


def summarize_k_sweep(args: argparse.Namespace) -> None:
    data_config = load_data_config(args.config)
    lambda_config = load_lambda_configuration(args.lambda_config)
    validate_lambda_scope(lambda_config)
    manifests = load_manifests(
        args.input_root,
        phase="k_sweep",
        lambda_config=lambda_config,
    )

    summary_frames = []
    seed_frames = []
    manifest_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for manifest_path, manifest in manifests:
        dataset = manifest["dataset"]
        k = int(manifest["k"])
        key = (dataset, k)
        if key in manifest_by_key:
            raise ValueError(f"Duplicate completed K-sweep runs for {key}.")
        manifest_by_key[key] = manifest
        if k < args.k_min or k > args.k_max:
            raise ValueError(f"{manifest_path}: unexpected K={k}.")

        paths = manifest.get("paths", {})
        required_paths = (
            "raw_dmkcn",
            "latents_dmkcn",
            "dmkcn_manifest",
            "computation_timings",
            "resource_usage",
            "metrics_raw",
            "metrics_seed_means",
            "metrics_condition_summary",
        )
        missing_paths = [name for name in required_paths if name not in paths]
        if missing_paths:
            raise ValueError(f"{manifest_path}: missing paths {missing_paths}.")
        for name in required_paths:
            path = Path(paths[name])
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"{manifest_path}: {name} is absent or empty: {path}"
                )

        summary_path = Path(paths["metrics_condition_summary"])
        summary = read_csv(summary_path)
        require_columns(
            summary,
            [
                "experiment_id",
                "run_id",
                "dataset",
                "k",
                "model",
                "condition",
                "alpha",
                "beta",
                *LAMBDA_COLUMNS,
                "NMI_median",
                "ARI_median",
                "ACC_median",
                "n_seeds",
            ],
            summary_path,
        )
        if len(summary) != 1:
            raise ValueError(f"{summary_path}: expected one canonical summary row.")
        row = summary.iloc[0]
        expected_triplet = lambda_config.resolve(dataset)
        expected_alpha = float(data_config[dataset]["full_alpha"])
        expected_beta = float(data_config[dataset]["full_beta"])
        if (
            row["dataset"] != dataset
            or int(row["k"]) != k
            or row["model"] != MODEL
            or row["condition"] != "full"
            or int(row["n_seeds"]) != len(SEEDS)
            or not np.isclose(float(row["alpha"]), expected_alpha)
            or not np.isclose(float(row["beta"]), expected_beta)
            or row["lambda_config_id"] != expected_triplet.lambda_config_id
            or not np.isclose(float(row["lambda1"]), expected_triplet.lambda1)
            or not np.isclose(float(row["lambda2"]), expected_triplet.lambda2)
            or not np.isclose(float(row["lambda3"]), expected_triplet.lambda3)
        ):
            raise ValueError(f"{summary_path}: canonical run identity mismatch.")
        summary["source_manifest"] = str(manifest_path)
        summary["latent_artifact"] = paths["latents_dmkcn"]
        summary["source_summary_file"] = str(summary_path)
        summary_frames.append(summary)

        seed_path = Path(paths["metrics_seed_means"])
        seed_frame = read_csv(seed_path)
        require_columns(
            seed_frame,
            [
                "experiment_id",
                "run_id",
                "dataset",
                "k",
                "model",
                "seed",
                "alpha",
                "beta",
                *LAMBDA_COLUMNS,
                *METRICS,
            ],
            seed_path,
        )
        if (
            set(seed_frame["dataset"]) != {dataset}
            or set(seed_frame["k"].astype(int)) != {k}
            or set(seed_frame["model"]) != {MODEL}
            or set(seed_frame["lambda_config_id"])
            != {expected_triplet.lambda_config_id}
            or not np.isclose(
                seed_frame["lambda1"].astype(float), expected_triplet.lambda1
            ).all()
            or not np.isclose(
                seed_frame["lambda2"].astype(float), expected_triplet.lambda2
            ).all()
            or not np.isclose(
                seed_frame["lambda3"].astype(float), expected_triplet.lambda3
            ).all()
        ):
            raise ValueError(f"{seed_path}: per-seed run identity mismatch.")
        validate_expected_seeds(
            seed_frame,
            ["dataset", "k", "model", *LAMBDA_COLUMNS, "alpha", "beta"],
            seed_path,
        )
        seed_frame["source_manifest"] = str(manifest_path)
        seed_frames.append(seed_frame)

        raw_path = Path(paths["metrics_raw"])
        raw_metrics = read_csv(raw_path)
        require_columns(
            raw_metrics,
            [
                "dataset",
                "k",
                "model",
                "seed",
                "repeat",
                "alpha",
                "beta",
                *LAMBDA_COLUMNS,
                *METRICS,
            ],
            raw_path,
        )
        duplicate_key = [
            "dataset",
            "k",
            "model",
            *LAMBDA_COLUMNS,
            "alpha",
            "beta",
            "seed",
            "repeat",
        ]
        if raw_metrics.duplicated(duplicate_key).any():
            raise ValueError(f"{raw_path}: duplicate raw metric rows detected.")
        validate_expected_seeds(
            raw_metrics,
            ["dataset", "k", "model", *LAMBDA_COLUMNS, "alpha", "beta"],
            raw_path,
            rows_per_seed=REPEAT,
        )
        read_csv(Path(paths["computation_timings"]))
        dmkcn_manifest = read_json(Path(paths["dmkcn_manifest"]))
        dmkcn_lambdas = dmkcn_manifest.get("lambda_configuration", {})
        if dmkcn_lambdas.get("lambda_config_id") != expected_triplet.lambda_config_id:
            raise ValueError(
                f"{paths['dmkcn_manifest']}: lambda configuration is not traceable."
            )

    scores = pd.concat(summary_frames, ignore_index=True)
    seed_scores = pd.concat(seed_frames, ignore_index=True)
    expected_keys = {
        (dataset, k)
        for dataset in DATASETS
        for k in range(args.k_min, args.k_max + 1)
    }
    observed_keys = set(zip(scores["dataset"], scores["k"].astype(int)))
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    coverage_rows = []
    for dataset in DATASETS:
        observed_k = sorted(
            int(value)
            for value in scores.loc[
                scores["dataset"] == dataset, "k"
            ].astype(int).unique()
        )
        missing_k = sorted(set(range(args.k_min, args.k_max + 1)) - set(observed_k))
        coverage_rows.append(
            {
                "dataset": dataset,
                "model": MODEL,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "n_expected": args.k_max - args.k_min + 1,
                "n_observed": len(observed_k),
                "observed_k": json.dumps(observed_k),
                "missing_k": json.dumps(missing_k),
                "complete": not missing_k,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    if missing or unexpected or len(scores) != len(expected_keys):
        raise ValueError(
            "K-sweep coverage mismatch. "
            f"Missing={missing}; unexpected={unexpected}; rows={len(scores)}."
        )

    ranked = scores.sort_values(
        [
            "dataset",
            "NMI_median",
            "ARI_median",
            "ACC_median",
            "k",
        ],
        ascending=[True, False, False, False, True],
    ).copy()
    ranked["nmi_rank"] = ranked.groupby(
        ["dataset", *LAMBDA_COLUMNS], dropna=False
    ).cumcount() + 1
    best = ranked.loc[ranked["nmi_rank"] == 1].copy()
    if len(best) != len(DATASETS):
        raise ValueError("Expected exactly one selected K per campaign dataset.")
    best["selection_metric"] = "NMI_median"
    best["tie_break_rule"] = "ARI_median,ACC_median,lower_k"
    if not best["latent_artifact"].map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("At least one selected latent artifact is missing.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(
        scores.sort_values(["dataset", "k"]),
        args.outdir / "k_sweep_condition_summary.csv",
    )
    write_csv_atomic(
        seed_scores.sort_values(["dataset", "k", "seed"]),
        args.outdir / "k_sweep_seed_scores.csv",
    )
    write_csv_atomic(coverage, args.outdir / "k_sweep_coverage.csv")
    write_csv_atomic(
        best.sort_values("dataset"),
        args.outdir / "best_k_by_nmi.csv",
    )
    write_json_atomic(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "phase": "k_sweep",
            "input_root": str(args.input_root),
            "lambda_config": str(args.lambda_config),
            "datasets": list(DATASETS),
            "k_range": [args.k_min, args.k_max],
            "seeds": list(SEEDS),
            "repeat": REPEAT,
            "n_completed_tasks": len(scores),
            "selection_metric": "NMI_median",
            "tie_break_rule": ["ARI_median", "ACC_median", "lower_k"],
        },
        args.outdir / "k_sweep_manifest.json",
    )
    print(best[["dataset", "k", *LAMBDA_COLUMNS, "NMI_median"]].to_string(index=False))


def summarize_alpha_beta(args: argparse.Namespace) -> None:
    lambda_config = load_lambda_configuration(args.lambda_config)
    validate_lambda_scope(lambda_config)
    best_k = read_csv(args.best_k)
    require_columns(
        best_k,
        ["dataset", "k", "model", "latent_artifact", *LAMBDA_COLUMNS],
        args.best_k,
    )
    best_k = best_k.loc[
        (best_k["dataset"].isin(DATASETS)) & (best_k["model"] == MODEL)
    ].copy()
    if len(best_k) != len(DATASETS) or set(best_k["dataset"]) != set(DATASETS):
        raise ValueError(
            "Selected-K table must contain exactly Darmanis, Li and Manno."
        )

    manifests = load_manifests(
        args.input_root,
        phase="alpha_beta",
        lambda_config=lambda_config,
    )
    if len(manifests) != len(DATASETS):
        raise ValueError(
            f"Expected {len(DATASETS)} completed alpha/beta tasks, got {len(manifests)}."
        )

    frames = []
    observed_datasets = set()
    for manifest_path, manifest in manifests:
        dataset = manifest["dataset"]
        if dataset in observed_datasets:
            raise ValueError(f"Duplicate completed alpha/beta task for {dataset}.")
        observed_datasets.add(dataset)
        selected = best_k.loc[best_k["dataset"] == dataset].iloc[0]
        if int(manifest["k"]) != int(selected["k"]):
            raise ValueError(
                f"{manifest_path}: K={manifest['k']} does not match selected "
                f"K={selected['k']}."
            )
        if (
            manifest.get("source_latent_artifact") != selected["latent_artifact"]
            or manifest.get("source_experiment_id") != selected["experiment_id"]
            or manifest.get("source_run_id") != selected["run_id"]
        ):
            raise ValueError(
                f"{manifest_path}: selected-K source identity does not match "
                f"{args.best_k}."
            )
        paths = manifest.get("paths", {})
        if "grid_tsv" not in paths:
            raise ValueError(f"{manifest_path}: missing grid_tsv path.")
        grid_path = Path(paths["grid_tsv"])
        raw = read_csv(grid_path, sep="\t")
        require_columns(
            raw,
            [
                "experiment_id",
                "run_id",
                "dataset",
                "k",
                "arm",
                "seed",
                "repeat",
                "method",
                "alpha",
                "beta",
                "c_mode",
                "d_mode",
                "nmi",
                "ari",
                "acc",
            ],
            grid_path,
        )
        if (
            set(raw["dataset"]) != {dataset}
            or set(raw["arm"]) != {ARM}
            or set(raw["k"].astype(int)) != {int(selected["k"])}
            or set(raw["method"]) != {"C_full_D_spectral"}
            or set(raw["c_mode"]) != {"full"}
            or set(raw["d_mode"]) != {"spectral_precomputed"}
        ):
            raise ValueError(f"{grid_path}: alpha/beta run identity mismatch.")
        observed_pairs = set(zip(raw["alpha"].astype(float), raw["beta"].astype(float)))
        if observed_pairs != set(DEFAULT_GRID):
            raise ValueError(
                f"{grid_path}: alpha/beta coverage mismatch; "
                f"missing={sorted(set(DEFAULT_GRID) - observed_pairs)}."
            )
        repeat_values = set(raw["repeat"].astype(int))
        if repeat_values != set(range(REPEAT)):
            raise ValueError(
                f"{grid_path}: expected repeats 0..{REPEAT - 1}, got {repeat_values}."
            )
        validate_expected_seeds(
            raw,
            ["dataset", "k", "arm", "alpha", "beta"],
            grid_path,
            rows_per_seed=REPEAT,
        )
        duplicate_key = ["dataset", "k", "arm", "alpha", "beta", "seed", "repeat"]
        if raw.duplicated(duplicate_key).any():
            raise ValueError(f"{grid_path}: duplicate experimental rows detected.")
        if len(raw) != len(DEFAULT_GRID) * len(SEEDS) * REPEAT:
            raise ValueError(
                f"{grid_path}: expected {len(DEFAULT_GRID) * len(SEEDS) * REPEAT} "
                f"rows, got {len(raw)}."
            )
        triplet = lambda_config.resolve(dataset)
        raw["model"] = MODEL
        raw["lambda_config_id"] = triplet.lambda_config_id
        raw["lambda1"] = triplet.lambda1
        raw["lambda2"] = triplet.lambda2
        raw["lambda3"] = triplet.lambda3
        raw["source_manifest"] = str(manifest_path)
        raw["source_file"] = str(grid_path)
        frames.append(raw)

    raw = pd.concat(frames, ignore_index=True)
    keys = [
        "experiment_id",
        "run_id",
        "dataset",
        "k",
        "model",
        *LAMBDA_COLUMNS,
        "alpha",
        "beta",
    ]
    seed_means = (
        raw.groupby(keys + ["seed"], dropna=False)[["nmi", "ari", "acc"]]
        .mean()
        .reset_index()
        .rename(columns={"nmi": "NMI", "ari": "ARI", "acc": "ACC"})
    )
    validate_expected_seeds(
        seed_means,
        ["dataset", "k", "model", *LAMBDA_COLUMNS, "alpha", "beta"],
        args.input_root,
    )
    summary = (
        seed_means.groupby(keys, dropna=False)[list(METRICS)]
        .agg(["median", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    counts = seed_means.groupby(keys, dropna=False).size().reset_index(name="n_seeds")
    summary = summary.merge(counts, on=keys, how="left")
    grid_order = {pair: index for index, pair in enumerate(DEFAULT_GRID)}
    summary["grid_order"] = [
        grid_order[(float(alpha), float(beta))]
        for alpha, beta in zip(summary["alpha"], summary["beta"])
    ]
    ranked = summary.sort_values(
        ["dataset", "NMI_median", "ARI_median", "ACC_median", "grid_order"],
        ascending=[True, False, False, False, True],
    ).copy()
    ranked["nmi_rank"] = ranked.groupby(
        ["dataset", *LAMBDA_COLUMNS], dropna=False
    ).cumcount() + 1
    best = ranked.loc[ranked["nmi_rank"] == 1].copy()
    best["selection_metric"] = "NMI_median"
    best["tie_break_rule"] = "ARI_median,ACC_median,grid_order"
    best = best.merge(
        best_k[["dataset", "latent_artifact"]],
        on="dataset",
        how="left",
        validate="one_to_one",
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(
        raw.sort_values(["dataset", "alpha", "beta", "seed", "repeat"]),
        args.outdir / "alpha_beta_raw.csv",
    )
    write_csv_atomic(
        seed_means.sort_values(["dataset", "alpha", "beta", "seed"]),
        args.outdir / "alpha_beta_seed_means.csv",
    )
    write_csv_atomic(
        summary.sort_values(["dataset", "grid_order"]),
        args.outdir / "alpha_beta_condition_summary.csv",
    )
    write_csv_atomic(
        best.sort_values("dataset"),
        args.outdir / "best_alpha_beta_by_nmi.csv",
    )
    write_json_atomic(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "phase": "alpha_beta",
            "input_root": str(args.input_root),
            "best_k": str(args.best_k),
            "lambda_config": str(args.lambda_config),
            "datasets": list(DATASETS),
            "alpha_beta_grid": [list(pair) for pair in DEFAULT_GRID],
            "seeds": list(SEEDS),
            "repeat": REPEAT,
            "n_completed_tasks": len(manifests),
            "selection_metric": "NMI_median",
            "tie_break_rule": ["ARI_median", "ACC_median", "grid_order"],
        },
        args.outdir / "alpha_beta_manifest.json",
    )
    print(best[["dataset", "k", "alpha", "beta", "NMI_median"]].to_string(index=False))


def select_seed_rows(scores: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in selected.itertuples(index=False):
        mask = (
            (scores["dataset"] == row.dataset)
            & (scores["model"] == row.model)
            & (scores["k"].astype(int) == int(row.k))
            & np.isclose(scores["alpha"].astype(float), float(row.alpha))
            & np.isclose(scores["beta"].astype(float), float(row.beta))
        )
        frame = scores.loc[mask].copy()
        if len(frame) != len(SEEDS) or set(frame["seed"].astype(int)) != set(SEEDS):
            raise ValueError(
                f"Expected ten selected seed rows for {row.dataset}, {row.model}, "
                f"K={row.k}, alpha={row.alpha}, beta={row.beta}; got {len(frame)}."
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def compare_with_campaign1(args: argparse.Namespace) -> None:
    new_best = read_csv(args.new_best)
    new_seed_scores = read_csv(args.new_seed_scores)
    reference_best = read_csv(args.reference_best)
    reference_seed_scores = read_csv(args.reference_seed_scores)
    required_best = ["dataset", "model", "k", "alpha", "beta"]
    required_scores = required_best + ["seed", *METRICS]
    require_columns(new_best, required_best + list(LAMBDA_COLUMNS), args.new_best)
    require_columns(new_seed_scores, required_scores + list(LAMBDA_COLUMNS), args.new_seed_scores)
    require_columns(reference_best, required_best, args.reference_best)
    require_columns(reference_seed_scores, required_scores, args.reference_seed_scores)

    new_best = new_best.loc[new_best["dataset"].isin(DATASETS)].copy()
    if len(new_best) != len(DATASETS) or set(new_best["dataset"]) != set(DATASETS):
        raise ValueError(
            "New selected table must contain exactly Darmanis, Li and Manno."
        )
    new_selected = select_seed_rows(new_seed_scores, new_best)

    summary_rows = []
    test_rows = []
    for reference_model, comparison in (
        ("scMUG", "campaign2_DMKCN_minus_campaign1_scMUG"),
        ("scMUG-DMKCN", "campaign2_DMKCN_minus_campaign1_DMKCN"),
    ):
        selected_reference = reference_best.loc[
            reference_best["dataset"].isin(DATASETS)
            & (reference_best["model"] == reference_model)
        ].copy()
        if len(selected_reference) != len(DATASETS):
            raise ValueError(
                f"Campaign-1 selection is incomplete for {reference_model}."
            )
        reference_selected = select_seed_rows(
            reference_seed_scores,
            selected_reference,
        )
        for dataset in DATASETS:
            new_dataset = new_selected.loc[
                new_selected["dataset"] == dataset
            ].set_index("seed")
            reference_dataset = reference_selected.loc[
                reference_selected["dataset"] == dataset
            ].set_index("seed")
            if new_dataset.index.has_duplicates or reference_dataset.index.has_duplicates:
                raise ValueError(f"{dataset}: duplicated selected seed rows.")
            common = sorted(set(new_dataset.index).intersection(reference_dataset.index))
            if common != list(SEEDS):
                raise ValueError(
                    f"{dataset}: expected paired seeds {list(SEEDS)}, got {common}."
                )
            new_config = new_best.loc[new_best["dataset"] == dataset].iloc[0]
            old_config = selected_reference.loc[
                selected_reference["dataset"] == dataset
            ].iloc[0]
            summary_row = {
                "comparison": comparison,
                "dataset": dataset,
                "new_k": int(new_config["k"]),
                "new_alpha": float(new_config["alpha"]),
                "new_beta": float(new_config["beta"]),
                "lambda_config_id": new_config["lambda_config_id"],
                "lambda1": float(new_config["lambda1"]),
                "lambda2": float(new_config["lambda2"]),
                "lambda3": float(new_config["lambda3"]),
                "reference_model": reference_model,
                "reference_k": int(old_config["k"]),
                "reference_alpha": float(old_config["alpha"]),
                "reference_beta": float(old_config["beta"]),
            }
            for metric in METRICS:
                new_values = new_dataset.loc[common, metric].to_numpy(dtype=float)
                old_values = reference_dataset.loc[common, metric].to_numpy(dtype=float)
                differences = new_values - old_values
                new_median = float(np.median(new_values))
                old_median = float(np.median(old_values))
                summary_row[f"{metric}_median_campaign2_DMKCN"] = new_median
                summary_row[f"{metric}_median_reference"] = old_median
                summary_row[f"{metric}_median_delta"] = new_median - old_median
                if np.allclose(differences, 0):
                    statistic, p_value = np.nan, 1.0
                else:
                    statistic, p_value = wilcoxon(differences)
                test_rows.append(
                    {
                        "comparison": comparison,
                        "dataset": dataset,
                        "metric": metric,
                        "direction": "campaign2_DMKCN_minus_reference",
                        "n_paired_seeds": len(common),
                        "mean_difference": float(np.mean(differences)),
                        "median_difference": float(np.median(differences)),
                        "wilcoxon_statistic": statistic,
                        "p_value": float(p_value),
                    }
                )
            summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    tests["p_value_fdr_bh"] = tests.groupby(
        ["comparison", "metric"]
    )["p_value"].transform(lambda values: fdr_bh(values.to_numpy(dtype=float)))
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(summary, args.outdir / "optimized_median_comparison.csv")
    write_csv_atomic(
        tests.sort_values(["comparison", "metric", "dataset"]),
        args.outdir / "paired_seed_tests.csv",
    )
    write_json_atomic(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "new_best": str(args.new_best),
            "new_seed_scores": str(args.new_seed_scores),
            "reference_best": str(args.reference_best),
            "reference_seed_scores": str(args.reference_seed_scores),
            "datasets": list(DATASETS),
            "metrics": list(METRICS),
            "test": "two-sided paired Wilcoxon across matched seeds",
            "multiple_testing": (
                "Benjamini-Hochberg across datasets, separately by comparison "
                "and metric"
            ),
            "interpretation": (
                "Exploratory post-selection benchmark: K, alpha/beta and lambdas "
                "were selected using reference labels."
            ),
        },
        args.outdir / "final_comparison_manifest.json",
    )


def parse_elapsed_value(value: str) -> float:
    fields = value.strip().split(":")
    try:
        numbers = [float(field) for field in fields]
    except ValueError as exc:
        raise ValueError(f"Invalid resource elapsed value: {value!r}") from exc
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Invalid resource elapsed value: {value!r}")


def parse_resource_usage(path: Path) -> dict[str, float]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Resource-usage file is absent or empty: {path}")
    fields: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("User time (seconds):"):
                fields["user_cpu_seconds"] = float(line.rsplit(":", 1)[1])
            elif line.startswith("System time (seconds):"):
                fields["system_cpu_seconds"] = float(line.rsplit(":", 1)[1])
            elif line.startswith("Percent of CPU this job got:"):
                fields["cpu_percent"] = float(line.rsplit(":", 1)[1].rstrip("%"))
            elif line.startswith("Elapsed (wall clock) time"):
                fields["task_wall_seconds"] = parse_elapsed_value(
                    line.split("):", 1)[1]
                )
            elif line.startswith("Maximum resident set size (kbytes):"):
                fields["max_rss_mb_task"] = (
                    float(line.rsplit(":", 1)[1]) / 1024.0
                )
    required = {
        "user_cpu_seconds",
        "system_cpu_seconds",
        "cpu_percent",
        "task_wall_seconds",
        "max_rss_mb_task",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"{path}: missing resource-measurement fields {missing}.")
    fields["total_cpu_seconds"] = (
        fields["user_cpu_seconds"] + fields["system_cpu_seconds"]
    )
    return fields


def summarize_computation(args: argparse.Namespace) -> None:
    lambda_config = load_lambda_configuration(args.lambda_config)
    validate_lambda_scope(lambda_config)
    rows = []
    for phase, root in (
        ("k_sweep", args.k_input_root),
        ("alpha_beta", args.alpha_input_root),
    ):
        manifests = load_manifests(root, phase=phase, lambda_config=lambda_config)
        for manifest_path, manifest in manifests:
            resource_path = Path(manifest["paths"]["resource_usage"])
            row: dict[str, Any] = {
                "phase": phase,
                "experiment_id": manifest["experiment_id"],
                "run_id": manifest["run_id"],
                "dataset": manifest["dataset"],
                "k": int(manifest["k"]),
                **{
                    key: manifest["lambdas"][key]
                    for key in LAMBDA_COLUMNS
                },
                "allocated_cpus": int(
                    manifest.get("slurm", {}).get("cpus_per_task") or 0
                ),
                "source_manifest": str(manifest_path),
                "source_resource_usage": str(resource_path),
                **parse_resource_usage(resource_path),
                "scientific_pipeline_seconds": np.nan,
                "peak_gpu_allocated_mb": np.nan,
                "peak_gpu_reserved_mb": np.nan,
                "peak_rss_mb_recorder": np.nan,
            }
            if phase == "k_sweep":
                timing_path = Path(manifest["paths"]["computation_timings"])
                timing = read_csv(timing_path)
                require_columns(
                    timing,
                    [
                        "stage",
                        "elapsed_seconds",
                        "peak_gpu_allocated_mb",
                        "peak_gpu_reserved_mb",
                        "peak_rss_mb",
                    ],
                    timing_path,
                )
                canonical = timing.loc[
                    timing["stage"] == "canonical_pipeline_total"
                ]
                if len(canonical) != 1:
                    raise ValueError(
                        f"{timing_path}: expected one canonical_pipeline_total row."
                    )
                row["scientific_pipeline_seconds"] = float(
                    canonical.iloc[0]["elapsed_seconds"]
                )
                row["peak_gpu_allocated_mb"] = float(
                    pd.to_numeric(
                        timing["peak_gpu_allocated_mb"], errors="coerce"
                    ).max()
                )
                row["peak_gpu_reserved_mb"] = float(
                    pd.to_numeric(
                        timing["peak_gpu_reserved_mb"], errors="coerce"
                    ).max()
                )
                row["peak_rss_mb_recorder"] = float(
                    pd.to_numeric(timing["peak_rss_mb"], errors="raise").max()
                )
            rows.append(row)

    raw = pd.DataFrame(rows)
    expected_counts = {"k_sweep": len(DATASETS) * (K_MAX - K_MIN + 1), "alpha_beta": len(DATASETS)}
    observed_counts = raw.groupby("phase").size().to_dict()
    if observed_counts != expected_counts:
        raise ValueError(
            f"Computation coverage mismatch: expected {expected_counts}, "
            f"observed {observed_counts}."
        )
    group_keys = ["phase", "dataset", *LAMBDA_COLUMNS]
    summary = (
        raw.groupby(group_keys, dropna=False)
        .agg(
            n_tasks=("run_id", "nunique"),
            total_task_wall_seconds=("task_wall_seconds", "sum"),
            median_task_wall_seconds=("task_wall_seconds", "median"),
            max_task_wall_seconds=("task_wall_seconds", "max"),
            total_cpu_seconds=("total_cpu_seconds", "sum"),
            max_task_average_cpu_percent=("cpu_percent", "max"),
            max_rss_mb=("max_rss_mb_task", "max"),
            max_gpu_allocated_mb=("peak_gpu_allocated_mb", "max"),
            max_gpu_reserved_mb=("peak_gpu_reserved_mb", "max"),
        )
        .reset_index()
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(
        raw.sort_values(["phase", "dataset", "k"]),
        args.outdir / "task_resources.csv",
    )
    write_csv_atomic(
        summary.sort_values(["phase", "dataset"]),
        args.outdir / "computation_summary.csv",
    )
    write_json_atomic(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "k_input_root": str(args.k_input_root),
            "alpha_input_root": str(args.alpha_input_root),
            "lambda_config": str(args.lambda_config),
            "expected_task_counts": expected_counts,
            "resource_source": (
                "Python resource.getrusage(RUSAGE_CHILDREN) wrapper plus "
                "scMUG computation recorder"
            ),
        },
        args.outdir / "computation_manifest.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    k_parser = subparsers.add_parser("k-sweep")
    k_parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/dmkcn_lambda_campaign2/k_sweep"),
    )
    k_parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    k_parser.add_argument(
        "--lambda-config",
        type=Path,
        default=Path("configs/dmkcn_lambdas_triplets"),
    )
    k_parser.add_argument("--k-min", type=int, default=K_MIN)
    k_parser.add_argument("--k-max", type=int, default=K_MAX)
    k_parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reporting/dmkcn_lambda_campaign2/k_sweep"),
    )

    alpha_parser = subparsers.add_parser("alpha-beta")
    alpha_parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/dmkcn_lambda_campaign2/alpha_beta"),
    )
    alpha_parser.add_argument(
        "--best-k",
        type=Path,
        default=Path("reporting/dmkcn_lambda_campaign2/k_sweep/best_k_by_nmi.csv"),
    )
    alpha_parser.add_argument(
        "--lambda-config",
        type=Path,
        default=Path("configs/dmkcn_lambdas_triplets"),
    )
    alpha_parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reporting/dmkcn_lambda_campaign2/alpha_beta"),
    )

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument(
        "--new-best",
        type=Path,
        default=Path(
            "reporting/dmkcn_lambda_campaign2/alpha_beta/"
            "best_alpha_beta_by_nmi.csv"
        ),
    )
    compare_parser.add_argument(
        "--new-seed-scores",
        type=Path,
        default=Path(
            "reporting/dmkcn_lambda_campaign2/alpha_beta/"
            "alpha_beta_seed_means.csv"
        ),
    )
    compare_parser.add_argument(
        "--reference-best",
        type=Path,
        default=Path("publication/alpha_beta/alpha_beta_best_by_nmi.csv"),
    )
    compare_parser.add_argument(
        "--reference-seed-scores",
        type=Path,
        default=Path("publication/alpha_beta/alpha_beta_seed_means.csv"),
    )
    compare_parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reporting/dmkcn_lambda_campaign2/final_comparison"),
    )

    computation_parser = subparsers.add_parser("computation")
    computation_parser.add_argument(
        "--k-input-root",
        type=Path,
        default=Path("outputs/dmkcn_lambda_campaign2/k_sweep"),
    )
    computation_parser.add_argument(
        "--alpha-input-root",
        type=Path,
        default=Path("outputs/dmkcn_lambda_campaign2/alpha_beta"),
    )
    computation_parser.add_argument(
        "--lambda-config",
        type=Path,
        default=Path("configs/dmkcn_lambdas_triplets"),
    )
    computation_parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reporting/dmkcn_lambda_campaign2/computation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "k-sweep":
        summarize_k_sweep(args)
    elif args.command == "alpha-beta":
        summarize_alpha_beta(args)
    elif args.command == "compare":
        compare_with_campaign1(args)
    else:
        summarize_computation(args)


if __name__ == "__main__":
    main()
