#!/usr/bin/env python3
"""Resolve the Baron/Darmanis/Li/Manno campaign-2 tasks without touching results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import yaml

from dmkcn.lambda_config import LambdaConfiguration, load_lambda_configuration
from read_configs import bash_export, ordered_datasets


CAMPAIGN_DATASETS = ("baron", "darmanis", "li", "manno")
K_MIN = 5
K_MAX = 20
MODEL = "scMUG-DMKCN"
ARM = "dmkcn"


def load_data_config(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    ordered_datasets(config)
    missing = sorted(set(CAMPAIGN_DATASETS).difference(config))
    if missing:
        raise ValueError(f"Campaign datasets absent from {path}: {missing}")
    return config


def validate_campaign_config(config: LambdaConfiguration) -> None:
    observed = tuple(config.datasets)
    if observed != CAMPAIGN_DATASETS:
        raise ValueError(
            "Campaign 2 is frozen to Baron, Darmanis, Li and Manno in "
            "alphabetical order; "
            f"observed {observed} in {config.source_file}."
        )
    if config.fallback != "error":
        raise ValueError("Campaign 2 requires fallback: error.")


def build_k_sweep_tasks(
    data_config_path: Path,
    lambda_config_path: Path,
    *,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> list[dict[str, Any]]:
    if k_min < 1 or k_max < k_min:
        raise ValueError(f"Invalid K range: {k_min}-{k_max}.")
    data_config = load_data_config(data_config_path)
    all_datasets = ordered_datasets(data_config)
    lambda_config = load_lambda_configuration(lambda_config_path)
    validate_campaign_config(lambda_config)

    tasks = []
    for dataset in CAMPAIGN_DATASETS:
        triplet = lambda_config.resolve(dataset)
        config_index = all_datasets.index(dataset) + 1
        for k in range(k_min, k_max + 1):
            tasks.append(
                {
                    "task_id": len(tasks),
                    "dataset": dataset,
                    "config_index": config_index,
                    "k": k,
                    **triplet.as_dict(),
                }
            )
    return tasks


def load_best_k_tasks(
    best_k_path: Path,
    data_config_path: Path,
    lambda_config_path: Path,
    *,
    check_artifacts: bool,
) -> list[dict[str, Any]]:
    data_config = load_data_config(data_config_path)
    all_datasets = ordered_datasets(data_config)
    lambda_config = load_lambda_configuration(lambda_config_path)
    validate_campaign_config(lambda_config)
    if not best_k_path.is_file():
        raise FileNotFoundError(f"Selected-K table not found: {best_k_path}")

    with best_k_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "experiment_id",
        "run_id",
        "dataset",
        "k",
        "model",
        "lambda_config_id",
        "lambda1",
        "lambda2",
        "lambda3",
        "latent_artifact",
    }
    if not rows:
        raise ValueError(f"Selected-K table is empty: {best_k_path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"{best_k_path} is missing required columns: {sorted(missing)}"
        )

    selected = [
        row
        for row in rows
        if row["dataset"] in CAMPAIGN_DATASETS and row["model"] == MODEL
    ]
    observed = [row["dataset"] for row in selected]
    if sorted(observed) != list(CAMPAIGN_DATASETS) or len(set(observed)) != len(
        CAMPAIGN_DATASETS
    ):
        raise ValueError(
            "Selected-K table must contain exactly one scMUG-DMKCN row for "
            f"Baron, Darmanis, Li and Manno; observed {observed}."
        )

    by_dataset = {row["dataset"]: row for row in selected}
    tasks = []
    for dataset in CAMPAIGN_DATASETS:
        row = by_dataset[dataset]
        triplet = lambda_config.resolve(dataset)
        for key in ("lambda1", "lambda2", "lambda3"):
            if not math.isclose(
                float(row[key]),
                getattr(triplet, key),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{dataset}: selected-K {key}={row[key]} does not match "
                    f"{triplet.as_dict()[key]} from {lambda_config_path}."
                )
        if row["lambda_config_id"] != triplet.lambda_config_id:
            raise ValueError(
                f"{dataset}: selected-K lambda_config_id={row['lambda_config_id']!r} "
                f"does not match {triplet.lambda_config_id!r}."
            )
        latent = Path(row["latent_artifact"])
        if check_artifacts and (not latent.is_file() or latent.stat().st_size == 0):
            raise FileNotFoundError(
                f"{dataset}: selected latent artifact is absent or empty: {latent}"
            )
        k = int(row["k"])
        if k < K_MIN or k > K_MAX:
            raise ValueError(f"{dataset}: selected K={k} is outside {K_MIN}-{K_MAX}.")
        tasks.append(
            {
                "task_id": len(tasks),
                "dataset": dataset,
                "config_index": all_datasets.index(dataset) + 1,
                "k": k,
                "model": MODEL,
                "arm": ARM,
                "latent_artifact": str(latent),
                "source_experiment_id": row["experiment_id"],
                "source_run_id": row["run_id"],
                **triplet.as_dict(),
            }
        )
    return tasks


def select_task(tasks: list[dict[str, Any]], task_id: int) -> dict[str, Any]:
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Task ID must be between 0 and {len(tasks) - 1}: {task_id}")
    return tasks[task_id]


def print_table(tasks: list[dict[str, Any]]) -> None:
    columns = ("task_id", "dataset", "k", "lambda_config_id", "lambda1", "lambda2", "lambda3")
    print("\t".join(columns))
    for task in tasks:
        print("\t".join(str(task[column]) for column in columns))


def export_task(task: dict[str, Any], task_count: int) -> None:
    bash_export("CAMPAIGN_TASK_COUNT", task_count)
    bash_export("CONFIG_INDEX", task["config_index"])
    bash_export("DATASET", task["dataset"])
    bash_export("CLUSTER_NUMBER", task["k"])
    bash_export("LAMBDA_CONFIG_ID", task["lambda_config_id"])
    bash_export("LAMBDA1", task["lambda1"])
    bash_export("LAMBDA2", task["lambda2"])
    bash_export("LAMBDA3", task["lambda3"])
    bash_export("LAMBDA_CAMPAIGN_ID", task["campaign_id"])
    bash_export("LAMBDA_SOURCE_FILE", task["source_file"])
    if "latent_artifact" in task:
        bash_export("MODEL", task["model"])
        bash_export("ARM", task["arm"])
        bash_export("LATENT_ARTIFACT", task["latent_artifact"])
        bash_export("SOURCE_EXPERIMENT_ID", task["source_experiment_id"])
        bash_export("SOURCE_RUN_ID", task["source_run_id"])


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--lambda-config",
        type=Path,
        default=Path("configs/dmkcn_lambdas_triplets"),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-id", type=int)
    selection.add_argument("--list", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    k_parser = subparsers.add_parser("k-sweep")
    add_common_arguments(k_parser)
    k_parser.add_argument("--k-min", type=int, default=K_MIN)
    k_parser.add_argument("--k-max", type=int, default=K_MAX)

    alpha_parser = subparsers.add_parser("alpha-beta")
    add_common_arguments(alpha_parser)
    alpha_parser.add_argument("--best-k", type=Path, required=True)
    alpha_parser.add_argument("--no-check-artifact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "k-sweep":
        tasks = build_k_sweep_tasks(
            args.config,
            args.lambda_config,
            k_min=args.k_min,
            k_max=args.k_max,
        )
    else:
        tasks = load_best_k_tasks(
            args.best_k,
            args.config,
            args.lambda_config,
            check_artifacts=not args.no_check_artifact,
        )
    if args.list:
        print_table(tasks)
        return
    export_task(select_task(tasks, args.task_id), len(tasks))


if __name__ == "__main__":
    main()
