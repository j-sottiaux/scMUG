"""Resolve alpha/beta-grid array tasks from the selected k table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from read_configs import bash_export, ordered_datasets
from summarize_k_sweep import load_dataset_config


PIPELINE_ORDER = ("scMUG", "scMUG-DMKCN")
ARM_BY_PIPELINE = {"scMUG": "autoencoder", "scMUG-DMKCN": "dmkcn"}
TAG_BY_PIPELINE = {"scMUG": "ae", "scMUG-DMKCN": "dmkcn"}
REQUIRED_COLUMNS = {
    "experiment_id",
    "run_id",
    "dataset",
    "k",
    "model",
    "latent_artifact",
}


def load_alpha_beta_tasks(best_k_path: Path, config_path: Path) -> pd.DataFrame:
    config = load_dataset_config(config_path)
    datasets = ordered_datasets(config)
    table = pd.read_csv(best_k_path)
    missing = REQUIRED_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"{best_k_path} is missing columns: {sorted(missing)}")

    table = table.loc[table["model"].isin(PIPELINE_ORDER)].copy()
    table["k"] = pd.to_numeric(table["k"], errors="raise").astype(int)
    expected = {(dataset, model) for dataset in datasets for model in PIPELINE_ORDER}
    observed = set(zip(table["dataset"], table["model"]))
    if observed != expected or len(table) != len(expected):
        raise ValueError(
            "Selected-k table must contain exactly one row per dataset and pipeline. "
            f"Missing: {sorted(expected - observed)}; unexpected: "
            f"{sorted(observed - expected)}"
        )

    dataset_order = {dataset: index for index, dataset in enumerate(datasets)}
    pipeline_order = {model: index for index, model in enumerate(PIPELINE_ORDER)}
    table["config_index"] = table["dataset"].map(dataset_order) + 1
    table["_dataset_order"] = table["dataset"].map(dataset_order)
    table["_pipeline_order"] = table["model"].map(pipeline_order)
    table = table.sort_values(["_dataset_order", "_pipeline_order"]).reset_index(
        drop=True
    )
    table["task_id"] = table.index
    table["arm"] = table["model"].map(ARM_BY_PIPELINE)
    table["model_tag"] = table["model"].map(TAG_BY_PIPELINE)
    return table.drop(columns=["_dataset_order", "_pipeline_order"])


def select_task(
    tasks: pd.DataFrame, task_id: int, *, check_artifact: bool = True
) -> dict:
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Task ID must be between 0 and {len(tasks) - 1}: {task_id}")
    task = tasks.iloc[task_id].to_dict()
    latent_path = Path(str(task["latent_artifact"]))
    if check_artifact and not latent_path.is_file():
        raise FileNotFoundError(
            f"Selected latent artifact does not exist for task {task_id}: {latent_path}"
        )
    return task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-k", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--no-check-artifact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_alpha_beta_tasks(args.best_k, args.config)
    task = select_task(
        tasks, args.task_id, check_artifact=not args.no_check_artifact
    )
    bash_export("ALPHA_BETA_TASK_COUNT", len(tasks))
    bash_export("CONFIG_INDEX", int(task["config_index"]))
    bash_export("DATASET", task["dataset"])
    bash_export("MODEL", task["model"])
    bash_export("MODEL_TAG", task["model_tag"])
    bash_export("ARM", task["arm"])
    bash_export("SELECTED_K", int(task["k"]))
    bash_export("LATENT_ARTIFACT", task["latent_artifact"])
    bash_export("SOURCE_EXPERIMENT_ID", task["experiment_id"])
    bash_export("SOURCE_RUN_ID", task["run_id"])


if __name__ == "__main__":
    main()
