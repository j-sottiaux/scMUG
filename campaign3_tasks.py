#!/usr/bin/env python3
"""Resolve the frozen dataset x K-regime task map for k-fusion campaign 3."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml

from dmkcn.lambda_config import load_lambda_configuration
from read_configs import (
    bash_export,
    get_cutoffs,
    get_gfm_correlation_rule,
    ordered_datasets,
)


EXPECTED_DATASETS = ("darmanis", "li", "manno", "muraro")
EXPECTED_REGIMES = ("oracle", "historical")
EXPECTED_CONDITIONS = (
    "cd_all_gfm",
    "direct_mean_frobenius",
    "cd_single_gfm",
    "direct_single_gfm",
    "direct_mean_raw",
)
EXPECTED_TOP_LEVEL = {
    "schema_version",
    "campaign_id",
    "data_config",
    "lambda_config",
    "datasets",
    "protocol",
}
REGIME_KEYS = {"role", "k_train", "k_cluster", "alpha", "beta"}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return payload


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _weight(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}.")
    return result


def load_campaign(path: Path) -> dict[str, Any]:
    payload = _load_yaml(path)
    if set(payload) != EXPECTED_TOP_LEVEL:
        raise ValueError(
            f"{path} must contain exactly {sorted(EXPECTED_TOP_LEVEL)}; "
            f"observed {sorted(payload)}."
        )
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported campaign schema: {payload['schema_version']!r}")
    campaign_id = payload["campaign_id"]
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id must be a non-empty string.")

    datasets = payload["datasets"]
    if not isinstance(datasets, dict) or tuple(datasets) != EXPECTED_DATASETS:
        raise ValueError(
            "Campaign 3 datasets must be Darmanis, Li, Manno and Muraro in "
            f"alphabetical order; observed {tuple(datasets) if isinstance(datasets, dict) else datasets!r}."
        )
    for dataset, entry in datasets.items():
        if not isinstance(entry, dict) or set(entry) != {"lambda_config_id", "regimes"}:
            raise ValueError(
                f"{dataset}: expected lambda_config_id and regimes, got {entry!r}."
            )
        regimes = entry["regimes"]
        if not isinstance(regimes, dict) or tuple(regimes) != EXPECTED_REGIMES:
            raise ValueError(
                f"{dataset}: regimes must be ordered as {EXPECTED_REGIMES}, got "
                f"{tuple(regimes) if isinstance(regimes, dict) else regimes!r}."
            )
        for regime, values in regimes.items():
            if not isinstance(values, dict) or set(values) != REGIME_KEYS:
                raise ValueError(
                    f"{dataset}/{regime}: expected exactly {sorted(REGIME_KEYS)}."
                )
            k_train = _positive_int(f"{dataset}/{regime}/k_train", values["k_train"])
            k_cluster = _positive_int(
                f"{dataset}/{regime}/k_cluster", values["k_cluster"]
            )
            if k_train != k_cluster:
                raise ValueError(
                    f"{dataset}/{regime}: k_train={k_train} must equal "
                    f"k_cluster={k_cluster} in campaign 3."
                )
            alpha = _weight(f"{dataset}/{regime}/alpha", values["alpha"])
            beta = _weight(f"{dataset}/{regime}/beta", values["beta"])
            if alpha == 0 and beta == 0:
                raise ValueError(f"{dataset}/{regime}: alpha and beta cannot both be zero.")
            if values["role"] not in {"primary", "sensitivity"}:
                raise ValueError(f"{dataset}/{regime}: invalid role {values['role']!r}.")

    protocol = payload["protocol"]
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping.")
    if tuple(protocol.get("conditions", ())) != EXPECTED_CONDITIONS:
        raise ValueError(
            f"Campaign conditions must be exactly {EXPECTED_CONDITIONS}."
        )
    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 10:
        raise ValueError("protocol.seeds must contain exactly 10 seeds.")
    parsed_seeds = [_positive_int("seed", value) for value in seeds]
    if len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError("protocol.seeds contains duplicates.")
    _positive_int("protocol.repeats", protocol.get("repeats"))
    _positive_int("protocol.kmeans_times", protocol.get("kmeans_times"))
    _positive_int("protocol.n_neighbour", protocol.get("n_neighbour"))
    _positive_int(
        "protocol.dmkcn_projection_dimension",
        protocol.get("dmkcn_projection_dimension"),
    )
    if protocol.get("primary_condition") != "direct_mean_frobenius":
        raise ValueError("Unexpected primary condition.")
    if protocol.get("reference_condition") != "cd_all_gfm":
        raise ValueError("Unexpected reference condition.")
    if protocol.get("primary_metric") != "NMI":
        raise ValueError("Campaign 3 primary metric must be NMI.")
    if protocol.get("primary_regime") != "oracle":
        raise ValueError("Campaign 3 primary regime must be oracle.")
    if protocol.get("cross_gfm_normalization") != "off_diagonal_frobenius":
        raise ValueError("Unexpected cross-GFM normalization rule.")
    if protocol.get("affinity_nonnegativity") != "clip":
        raise ValueError("Unexpected affinity non-negativity rule.")
    if protocol.get("preserve_diagonal") is not True:
        raise ValueError("Campaign 3 must preserve the affinity diagonal.")
    return payload


def build_tasks(campaign_path: Path) -> list[dict[str, Any]]:
    campaign = load_campaign(campaign_path)
    project_root = campaign_path.resolve().parent.parent
    data_path = project_root / campaign["data_config"]
    lambda_path = project_root / campaign["lambda_config"]
    data_config = _load_yaml(data_path)
    all_data_names = ordered_datasets(data_config)
    lambda_config = load_lambda_configuration(lambda_path)
    protocol = campaign["protocol"]

    missing = sorted(set(EXPECTED_DATASETS).difference(data_config))
    if missing:
        raise ValueError(f"Datasets missing from {data_path}: {missing}")

    tasks: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        dataset_config = data_config[dataset]
        campaign_dataset = campaign["datasets"][dataset]
        candidate_id = campaign_dataset["lambda_config_id"]
        triplet = lambda_config.resolve_candidate(dataset, candidate_id)
        if int(dataset_config["ground_truth_k"]) != int(
            campaign_dataset["regimes"]["oracle"]["k_cluster"]
        ):
            raise ValueError(
                f"{dataset}: oracle K does not match ground_truth_k from {data_path}."
            )
        n_gfm = _positive_int(f"{dataset}/n_gfm", dataset_config["n_gfm"])
        cutoffs = get_cutoffs(dataset_config, n_gfm)
        correlation_rule = get_gfm_correlation_rule(dataset_config)
        for regime in EXPECTED_REGIMES:
            values = campaign_dataset["regimes"][regime]
            tasks.append(
                {
                    "task_id": len(tasks),
                    "campaign_id": campaign["campaign_id"],
                    "campaign_config": str(campaign_path),
                    "data_config": str(data_path),
                    "lambda_config": str(lambda_path),
                    "dataset": dataset,
                    "config_index": all_data_names.index(dataset) + 1,
                    "regime": regime,
                    "role": values["role"],
                    "k_train": int(values["k_train"]),
                    "k_cluster": int(values["k_cluster"]),
                    "alpha": float(values["alpha"]),
                    "beta": float(values["beta"]),
                    "n_gfm": n_gfm,
                    "cutoffs": cutoffs,
                    "gfm_correlation_rule": correlation_rule,
                    "allow_pseudo_counts": bool(
                        dataset_config.get("dmkcn_allow_pseudo_counts", False)
                    ),
                    "lambda_config_id": triplet.lambda_config_id,
                    "lambda1": triplet.lambda1,
                    "lambda2": triplet.lambda2,
                    "lambda3": triplet.lambda3,
                    "lambda_source_campaign_id": triplet.campaign_id,
                    "lambda_source_file": triplet.source_file,
                    "seeds": [int(value) for value in protocol["seeds"]],
                    "repeats": int(protocol["repeats"]),
                    "kmeans_times": int(protocol["kmeans_times"]),
                    "n_neighbour": int(protocol["n_neighbour"]),
                    "red_global": protocol["red_global"],
                    "red_local": protocol["red_local"],
                    "projection": protocol["dmkcn_projection"],
                    "projection_dimension": int(
                        protocol["dmkcn_projection_dimension"]
                    ),
                }
            )
    return tasks


def select_task(tasks: list[dict[str, Any]], task_id: int) -> dict[str, Any]:
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Task ID must be between 0 and {len(tasks) - 1}: {task_id}")
    return tasks[task_id]


def print_table(tasks: list[dict[str, Any]]) -> None:
    columns = (
        "task_id",
        "dataset",
        "regime",
        "role",
        "k_train",
        "k_cluster",
        "lambda_config_id",
        "lambda1",
        "lambda2",
        "lambda3",
        "alpha",
        "beta",
    )
    print("\t".join(columns))
    for task in tasks:
        print("\t".join(str(task[column]) for column in columns))


def export_task(task: dict[str, Any], task_count: int) -> None:
    values = {
        "CAMPAIGN_TASK_COUNT": task_count,
        "CAMPAIGN_ID": task["campaign_id"],
        "CAMPAIGN_CONFIG": task["campaign_config"],
        "DATA_CONFIG": task["data_config"],
        "LAMBDA_CONFIG": task["lambda_config"],
        "CONFIG_INDEX": task["config_index"],
        "DATASET": task["dataset"],
        "REGIME": task["regime"],
        "REGIME_ROLE": task["role"],
        "K_TRAIN": task["k_train"],
        "K_CLUSTER": task["k_cluster"],
        "ALPHA": task["alpha"],
        "BETA": task["beta"],
        "N_GFM": task["n_gfm"],
        "CUTOFFS": task["cutoffs"],
        "GFM_CORRELATION_RULE": task["gfm_correlation_rule"],
        "DMKCN_ALLOW_PSEUDO_COUNTS": str(task["allow_pseudo_counts"]).lower(),
        "LAMBDA_CONFIG_ID": task["lambda_config_id"],
        "LAMBDA1": task["lambda1"],
        "LAMBDA2": task["lambda2"],
        "LAMBDA3": task["lambda3"],
        "LAMBDA_SOURCE_CAMPAIGN_ID": task["lambda_source_campaign_id"],
        "LAMBDA_SOURCE_FILE": task["lambda_source_file"],
        "SEEDS": ",".join(str(value) for value in task["seeds"]),
        "REPEAT": task["repeats"],
        "KMEANS_TIMES": task["kmeans_times"],
        "N_NEIGHBOUR": task["n_neighbour"],
        "RED_GLOBAL": task["red_global"],
        "RED_LOCAL": task["red_local"],
        "DMKCN_PROJECTION": task["projection"],
        "DMKCN_PROJECTION_DIMENSION": task["projection_dimension"],
    }
    for name, value in values.items():
        bash_export(name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=Path("configs/dmkcn_kfusion_campaign3.yaml"),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-id", type=int)
    selection.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args.campaign_config)
    if args.list:
        print_table(tasks)
        return
    export_task(select_task(tasks, args.task_id), len(tasks))


if __name__ == "__main__":
    main()
