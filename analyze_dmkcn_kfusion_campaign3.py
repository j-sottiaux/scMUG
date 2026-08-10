#!/usr/bin/env python3
"""Analyze direct per-GFM and fused DMKCN K representations for campaign 3."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_rand_score

from ablation_block_c_d import (
    build_mat1,
    build_mat2,
    reduce_global_views,
    spectral_precomputed,
)
from dmkcn.integration import affinity_from_K
from utils import calc_acc, calc_ari, calc_nmi, set_seed


METRICS = ("NMI", "ARI", "ACC")


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Expected a non-empty list of unique integers.")
    return values


def sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def metric_values(y_true: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "NMI": float(calc_nmi(y_true, labels)),
        "ARI": float(calc_ari(y_true, labels)),
        "ACC": float(calc_acc(y_true, labels)),
    }


def off_diagonal_frobenius(matrix: np.ndarray) -> float:
    diagonal_sq = float(np.square(np.diag(matrix), dtype=np.float64).sum())
    total_sq = float(np.square(matrix, dtype=np.float64).sum())
    return float(math.sqrt(max(total_sq - diagonal_sq, 0.0)))


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, float]:
    array = np.asarray(matrix, dtype=np.float32)
    frobenius = float(np.linalg.norm(array))
    mean = float(array.mean())
    std = float(array.std())
    asymmetry = float(np.linalg.norm(array - array.T) / max(frobenius, 1e-12))
    diagonal = np.diag(array)
    return {
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": float(std / mean) if abs(mean) > 1e-12 else np.nan,
        "negative_fraction": float(np.mean(array < 0)),
        "frobenius_norm": frobenius,
        "off_diagonal_frobenius_norm": off_diagonal_frobenius(array),
        "diagonal_mean": float(diagonal.mean()),
        "diagonal_std": float(diagonal.std()),
        "asymmetry_relative_frobenius": asymmetry,
    }


def centered_alignment(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    raw_denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    raw = float(np.dot(x, y) / raw_denom) if raw_denom > 0 else np.nan
    x = x - x.mean()
    y = y - y.mean()
    centered_denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    centered = (
        float(np.dot(x, y) / centered_denom) if centered_denom > 0 else np.nan
    )
    return raw, centered


def normalized_affinity_eigenvalues(
    affinity: np.ndarray, n_clusters: int
) -> dict[str, Any]:
    array = np.asarray(affinity, dtype=np.float64)
    degrees = array.sum(axis=1)
    if np.any(~np.isfinite(degrees)) or np.any(degrees <= 0):
        return {"eigenvalues": [], "target_eigengap": np.nan, "status": "invalid_degree"}
    inverse_sqrt = 1.0 / np.sqrt(degrees)
    normalized = inverse_sqrt[:, None] * array * inverse_sqrt[None, :]
    count = min(max(n_clusters + 2, 12), normalized.shape[0] - 1)
    try:
        values = eigsh(normalized, k=count, which="LA", return_eigenvectors=False)
        values = np.sort(np.asarray(values, dtype=float))[::-1]
        gap = (
            float(values[n_clusters - 1] - values[n_clusters])
            if n_clusters < len(values)
            else np.nan
        )
        return {
            "eigenvalues": [float(value) for value in values],
            "target_eigengap": gap,
            "status": "ok",
        }
    except Exception as exc:  # diagnostic failure must remain visible
        return {
            "eigenvalues": [],
            "target_eigengap": np.nan,
            "status": f"error:{type(exc).__name__}:{exc}",
        }


def canonical_partition(labels: np.ndarray) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    canonical: list[int] = []
    for value in np.asarray(labels, dtype=int):
        item = int(value)
        if item not in mapping:
            mapping[item] = len(mapping)
        canonical.append(mapping[item])
    return tuple(canonical)


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    cell_metadata = pd.read_csv(
        args.cell_metadata,
        dtype={"cell_id": str, "cell_type": str},
    )
    required_cells = {"cell_position", "cell_id", "cell_type", "encoded_label"}
    missing = sorted(required_cells.difference(cell_metadata.columns))
    if missing:
        raise ValueError(f"Cell metadata missing columns: {missing}")
    if cell_metadata.empty or cell_metadata["cell_id"].duplicated().any():
        raise ValueError("Cell metadata must contain unique, non-empty cell IDs.")
    expected_positions = np.arange(len(cell_metadata))
    if not np.array_equal(cell_metadata["cell_position"].to_numpy(), expected_positions):
        raise ValueError("Cell positions are not contiguous and ordered from zero.")
    if cell_metadata[list(required_cells)].isna().any().any():
        raise ValueError("Cell metadata contains missing values.")
    cell_ids = cell_metadata["cell_id"].astype(str).tolist()
    cell_hash = sha256_lines(cell_ids)

    with open(args.gfm_membership, "r", encoding="utf-8") as handle:
        membership = json.load(handle)
    if int(membership["n_cells"]) != len(cell_metadata):
        raise ValueError("GFM membership and cell metadata disagree on n_cells.")
    if membership["cell_ids_sha256"] != cell_hash:
        raise ValueError("Cell identifier hash differs between metadata artifacts.")
    if int(membership["n_gfm"]) != args.n_gfm:
        raise ValueError("GFM membership has an unexpected number of GFMs.")
    if len(membership["gfm_membership"]) != args.n_gfm:
        raise ValueError("GFM membership list is incomplete.")

    latents = joblib.load(args.latents)
    if not isinstance(latents, (list, tuple)) or len(latents) != len(args.seeds):
        raise ValueError("Latent artifact must contain one entry per configured seed.")
    validated_latents = []
    for seed, latent in zip(args.seeds, latents):
        array = np.asarray(latent, dtype=np.float32)
        expected = (len(cell_metadata), args.n_gfm, args.projection_dimension)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(
                f"Seed {seed}: expected finite latent shape {expected}, got {array.shape}."
            )
        validated_latents.append(array)

    reference = joblib.load(args.reference_predictions)
    expected_reference = len(args.seeds) * args.repeats
    if not isinstance(reference, (list, tuple)) or len(reference) != expected_reference:
        raise ValueError(
            f"Expected {expected_reference} reference partitions, got "
            f"{len(reference) if isinstance(reference, (list, tuple)) else type(reference)}."
        )
    validated_reference = []
    for index, labels in enumerate(reference):
        array = np.asarray(labels, dtype=int)
        if array.shape != (len(cell_metadata),):
            raise ValueError(f"Reference partition {index} has shape {array.shape}.")
        validated_reference.append(array)

    reference_metrics = pd.read_csv(args.reference_metrics)
    required_metrics = {"seed", "repeat", *METRICS}
    missing_metrics = sorted(required_metrics.difference(reference_metrics.columns))
    if missing_metrics:
        raise ValueError(f"Reference metrics missing columns: {missing_metrics}")
    if len(reference_metrics) != expected_reference:
        raise ValueError(
            f"Expected {expected_reference} reference metric rows, got "
            f"{len(reference_metrics)}."
        )
    if reference_metrics.duplicated(["seed", "repeat"]).any():
        raise ValueError("Reference metrics contain duplicate seed/repeat rows.")
    reference_metrics = reference_metrics.set_index(["seed", "repeat"]).sort_index()

    return {
        "cell_metadata": cell_metadata,
        "cell_hash": cell_hash,
        "membership": membership,
        "latents": validated_latents,
        "reference": validated_reference,
        "reference_metrics": reference_metrics,
    }


def add_partition(
    rows: list[dict[str, Any]],
    partitions: list[np.ndarray],
    *,
    args: argparse.Namespace,
    y_true: np.ndarray,
    labels: np.ndarray,
    condition: str,
    seed: int,
    repeat_index: int,
    gfm_index: int | None,
    normalization: str,
) -> None:
    labels = np.asarray(labels, dtype=int)
    partition_id = len(partitions)
    row = {
        "partition_id": partition_id,
        "campaign_id": args.campaign_id,
        "campaign_instance_id": args.campaign_instance_id,
        "dataset": args.dataset,
        "regime": args.regime,
        "regime_role": args.regime_role,
        "k_train": args.k_train,
        "k_cluster": args.k_cluster,
        "lambda_config_id": args.lambda_config_id,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "lambda3": args.lambda3,
        "condition": condition,
        "seed": int(seed),
        "repeat": int(repeat_index),
        "gfm_index": gfm_index,
        "normalization": normalization,
        "alpha": args.alpha if condition.startswith("cd_") else np.nan,
        "beta": args.beta if condition.startswith("cd_") else np.nan,
        "n_predicted_clusters": int(np.unique(labels).size),
        **metric_values(y_true, labels),
    }
    rows.append(row)
    partitions.append(labels.astype(np.int32, copy=False))


def build_gfm_overlap(membership: dict[str, Any], args: argparse.Namespace):
    entries = membership["gfm_membership"]
    sets = {int(entry["gfm_index"]): set(entry["extended_genes"]) for entry in entries}
    pair_rows = []
    for left, right in itertools.combinations(sorted(sets), 2):
        intersection = sets[left] & sets[right]
        union = sets[left] | sets[right]
        pair_rows.append(
            {
                "dataset": args.dataset,
                "regime": args.regime,
                "gfm_left": left,
                "gfm_right": right,
                "n_left": len(sets[left]),
                "n_right": len(sets[right]),
                "n_intersection": len(intersection),
                "n_union": len(union),
                "jaccard": len(intersection) / len(union) if union else np.nan,
                "overlap_coefficient": (
                    len(intersection) / min(len(sets[left]), len(sets[right]))
                    if sets[left] and sets[right]
                    else np.nan
                ),
            }
        )
    counts: dict[str, int] = {}
    for genes in sets.values():
        for gene in genes:
            counts[gene] = counts.get(gene, 0) + 1
    membership_rows = [
        {"dataset": args.dataset, "regime": args.regime, "n_gfm_memberships": count}
        for count in counts.values()
    ]
    return pd.DataFrame(pair_rows), pd.DataFrame(membership_rows)


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    inputs = load_inputs(args)
    y_true = inputs["cell_metadata"]["encoded_label"].to_numpy(dtype=int)
    n_cells = len(y_true)
    rows: list[dict[str, Any]] = []
    partitions: list[np.ndarray] = []
    kernel_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    fused_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []

    reference_index = 0
    for seed_index, seed in enumerate(args.seeds):
        kernel_path = args.kernels_dir / f"seed{seed}_kernel_representations.npz"
        if not kernel_path.is_file() or kernel_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing kernel artifact: {kernel_path}")
        with np.load(kernel_path, allow_pickle=False) as archive:
            required = {"kernels", "gfm_indices", "seed", "cell_ids_sha256"}
            if not required.issubset(archive.files):
                raise ValueError(f"{kernel_path} lacks {sorted(required - set(archive.files))}.")
            kernels = np.asarray(archive["kernels"], dtype=np.float32)
            archive_seed = int(np.asarray(archive["seed"]).item())
            archive_hash = str(np.asarray(archive["cell_ids_sha256"]).item())
            gfm_indices = np.asarray(archive["gfm_indices"], dtype=int)
        expected_shape = (args.n_gfm, n_cells, n_cells)
        if kernels.shape != expected_shape or not np.isfinite(kernels).all():
            raise ValueError(
                f"Seed {seed}: expected finite kernels {expected_shape}, got {kernels.shape}."
            )
        if archive_seed != seed or archive_hash != inputs["cell_hash"]:
            raise ValueError(f"Seed/hash mismatch in {kernel_path}.")
        if not np.array_equal(gfm_indices, np.arange(1, args.n_gfm + 1)):
            raise ValueError(f"Unexpected GFM indices in {kernel_path}.")

        affinities = []
        norms = []
        for offset, kernel in enumerate(kernels):
            gfm_index = offset + 1
            affinity = affinity_from_K(kernel, nonneg="clip")
            norm = off_diagonal_frobenius(affinity)
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError(
                    f"Seed {seed}, GFM {gfm_index}: invalid off-diagonal norm {norm}."
                )
            affinities.append(affinity)
            norms.append(norm)
            kernel_rows.append(
                {
                    "dataset": args.dataset,
                    "regime": args.regime,
                    "seed": seed,
                    "gfm_index": gfm_index,
                    "matrix": "K",
                    **matrix_diagnostics(kernel),
                }
            )
            kernel_rows.append(
                {
                    "dataset": args.dataset,
                    "regime": args.regime,
                    "seed": seed,
                    "gfm_index": gfm_index,
                    "matrix": "A_sym_clip",
                    **matrix_diagnostics(affinity),
                }
            )

        for left, right in itertools.combinations(range(args.n_gfm), 2):
            raw_alignment, centered = centered_alignment(
                affinities[left], affinities[right]
            )
            pairwise_rows.append(
                {
                    "dataset": args.dataset,
                    "regime": args.regime,
                    "seed": seed,
                    "gfm_left": left + 1,
                    "gfm_right": right + 1,
                    "frobenius_alignment": raw_alignment,
                    "centered_frobenius_alignment": centered,
                }
            )

        raw_mean = np.mean(np.stack(affinities), axis=0, dtype=np.float64).astype(
            np.float32
        )
        normalized_mean = np.mean(
            np.stack(
                [affinity / norm for affinity, norm in zip(affinities, norms)]
            ),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        norm_ratio = float(max(norms) / min(norms))
        for name, matrix in (
            ("direct_mean_frobenius", normalized_mean),
            ("direct_mean_raw", raw_mean),
        ):
            fused_rows.append(
                {
                    "dataset": args.dataset,
                    "regime": args.regime,
                    "seed": seed,
                    "condition": name,
                    "gfm_norm_ratio_max_min": norm_ratio,
                    **matrix_diagnostics(matrix),
                }
            )
            spectrum = normalized_affinity_eigenvalues(matrix, args.k_cluster)
            spectrum_rows.append(
                {
                    "dataset": args.dataset,
                    "regime": args.regime,
                    "seed": seed,
                    "condition": name,
                    "k_cluster": args.k_cluster,
                    "target_eigengap": spectrum["target_eigengap"],
                    "status": spectrum["status"],
                    "eigenvalues": json.dumps(spectrum["eigenvalues"]),
                }
            )

        latent_all = inputs["latents"][seed_index]
        single_cd_cache: dict[int, tuple[np.ndarray, list[np.ndarray]]] = {}
        for gfm_index in range(1, args.n_gfm + 1):
            latent_single = latent_all[:, gfm_index - 1 : gfm_index, :]
            mat2 = build_mat2(latent_single, args.n_neighbour, args.red_local)
            global_views = reduce_global_views(latent_single, args.red_global)
            single_cd_cache[gfm_index] = (mat2, global_views)

        for repeat_index in range(args.repeats):
            repeat_seed = seed + repeat_index
            set_seed(repeat_seed)
            add_partition(
                rows,
                partitions,
                args=args,
                y_true=y_true,
                labels=inputs["reference"][reference_index],
                condition="cd_all_gfm",
                seed=seed,
                repeat_index=repeat_index,
                gfm_index=None,
                normalization="scmug_cd",
            )
            computed_reference = metric_values(
                y_true, inputs["reference"][reference_index]
            )
            expected_reference_metrics = inputs["reference_metrics"].loc[
                (seed, repeat_index)
            ]
            for metric in METRICS:
                if not np.isclose(
                    computed_reference[metric],
                    float(expected_reference_metrics[metric]),
                    rtol=0.0,
                    atol=1e-4,
                ):
                    raise ValueError(
                        f"Reference prediction/metric mismatch for seed={seed}, "
                        f"repeat={repeat_index}, metric={metric}: "
                        f"{computed_reference[metric]} != "
                        f"{expected_reference_metrics[metric]}."
                    )
            reference_index += 1

            for condition, matrix, normalization in (
                ("direct_mean_frobenius", normalized_mean, "off_diagonal_frobenius"),
                ("direct_mean_raw", raw_mean, "none_between_gfm"),
            ):
                labels = spectral_precomputed(matrix, args.k_cluster, repeat_seed)
                add_partition(
                    rows,
                    partitions,
                    args=args,
                    y_true=y_true,
                    labels=labels,
                    condition=condition,
                    seed=seed,
                    repeat_index=repeat_index,
                    gfm_index=None,
                    normalization=normalization,
                )

            for gfm_index, affinity in enumerate(affinities, start=1):
                direct_labels = spectral_precomputed(
                    affinity, args.k_cluster, repeat_seed
                )
                add_partition(
                    rows,
                    partitions,
                    args=args,
                    y_true=y_true,
                    labels=direct_labels,
                    condition="direct_single_gfm",
                    seed=seed,
                    repeat_index=repeat_index,
                    gfm_index=gfm_index,
                    normalization="single_affinity",
                )

                latent_single = latent_all[:, gfm_index - 1 : gfm_index, :]
                mat2, global_views = single_cd_cache[gfm_index]
                mat1 = build_mat1(
                    latent_single,
                    n_clusters=args.k_cluster,
                    kmeans_times=args.kmeans_times,
                    red_global=args.red_global,
                    thread_num=args.thread_num,
                    seed=seed,
                    repeat_index=repeat_index,
                    reduced_views=global_views,
                )
                cd_labels = spectral_precomputed(
                    args.alpha * mat1 + args.beta * mat2,
                    args.k_cluster,
                    repeat_seed,
                )
                add_partition(
                    rows,
                    partitions,
                    args=args,
                    y_true=y_true,
                    labels=cd_labels,
                    condition="cd_single_gfm",
                    seed=seed,
                    repeat_index=repeat_index,
                    gfm_index=gfm_index,
                    normalization="scmug_cd_single",
                )

    raw = pd.DataFrame(rows)
    expected_rows = len(args.seeds) * args.repeats * (3 + 2 * args.n_gfm)
    if len(raw) != expected_rows or len(partitions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} partitions, got {len(raw)} metrics and "
            f"{len(partitions)} label vectors."
        )

    group_without_repeat = [
        "campaign_id",
        "campaign_instance_id",
        "dataset",
        "regime",
        "regime_role",
        "k_train",
        "k_cluster",
        "lambda_config_id",
        "lambda1",
        "lambda2",
        "lambda3",
        "condition",
        "gfm_index",
        "normalization",
        "alpha",
        "beta",
        "seed",
    ]
    seed_means = (
        raw.groupby(group_without_repeat, dropna=False)[list(METRICS)]
        .mean()
        .reset_index()
    )
    summary_group = group_without_repeat[:-1]
    summary = (
        seed_means.groupby(summary_group, dropna=False)[list(METRICS)]
        .agg(["median", "mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    per_task_contrasts = []
    condition_by_seed: dict[str, pd.DataFrame] = {}
    for condition in (
        "cd_all_gfm",
        "direct_mean_frobenius",
        "direct_mean_raw",
        "direct_single_gfm",
        "cd_single_gfm",
    ):
        selected = seed_means[seed_means["condition"] == condition]
        if condition in {"direct_single_gfm", "cd_single_gfm"}:
            selected = selected.groupby("seed", as_index=False)[list(METRICS)].mean()
        condition_by_seed[condition] = selected.set_index("seed")
    contrast_pairs = (
        ("direct_mean_frobenius_minus_cd_all_gfm", "direct_mean_frobenius", "cd_all_gfm"),
        ("frobenius_mean_minus_raw_mean", "direct_mean_frobenius", "direct_mean_raw"),
        ("fused_direct_minus_mean_single_direct", "direct_mean_frobenius", "direct_single_gfm"),
    )
    for contrast, left_name, right_name in contrast_pairs:
        left = condition_by_seed[left_name]
        right = condition_by_seed[right_name]
        common = sorted(set(left.index) & set(right.index))
        if common != sorted(args.seeds):
            raise ValueError(f"{contrast}: incomplete paired seed set {common}.")
        for seed in common:
            for metric in METRICS:
                per_task_contrasts.append(
                    {
                        "dataset": args.dataset,
                        "regime": args.regime,
                        "contrast": contrast,
                        "seed": int(seed),
                        "metric": metric,
                        "left_value": float(left.loc[seed, metric]),
                        "right_value": float(right.loc[seed, metric]),
                        "difference": float(left.loc[seed, metric] - right.loc[seed, metric]),
                    }
                )

    partition_array = np.stack(partitions).astype(np.int32, copy=False)
    stability_rows = []
    for keys, frame in raw.groupby(
        ["condition", "gfm_index", "seed"], dropna=False
    ):
        indices = frame.sort_values("repeat")["partition_id"].to_numpy(dtype=int)
        labels_group = partition_array[indices]
        pairwise = [
            adjusted_rand_score(labels_group[left], labels_group[right])
            for left, right in itertools.combinations(range(len(labels_group)), 2)
        ]
        canonical = {canonical_partition(labels) for labels in labels_group}
        stability_rows.append(
            {
                "dataset": args.dataset,
                "regime": args.regime,
                "condition": keys[0],
                "gfm_index": keys[1],
                "seed": int(keys[2]),
                "n_distinct_partitions": len(canonical),
                "pairwise_ari_mean": float(np.mean(pairwise)) if pairwise else np.nan,
                "pairwise_ari_min": float(np.min(pairwise)) if pairwise else np.nan,
            }
        )

    li_sensitivity = pd.DataFrame()
    if args.dataset == "li":
        merged_labels = np.asarray(
            [re.sub(r"_B[12]$", "", value) for value in inputs["cell_metadata"]["cell_type"].astype(str)]
        )
        _, merged_y = np.unique(merged_labels, return_inverse=True)
        sensitivity_rows = []
        for row, labels in zip(rows, partition_array):
            sensitivity_rows.append(
                {
                    **{key: row[key] for key in ("partition_id", "dataset", "regime", "condition", "seed", "repeat", "gfm_index")},
                    "label_scheme": "strip_terminal_B1_B2",
                    "n_reference_classes": int(np.unique(merged_y).size),
                    **metric_values(merged_y, labels),
                }
            )
        li_sensitivity = pd.DataFrame(sensitivity_rows)

    gfm_overlap, gfm_membership_summary = build_gfm_overlap(
        inputs["membership"], args
    )
    output_root = args.output_root
    prefix = args.output_prefix
    paths = {
        "raw_metrics": output_root / "results" / f"{prefix}_raw_metrics.csv",
        "seed_means": output_root / "results" / f"{prefix}_seed_means.csv",
        "condition_summary": output_root / "results" / f"{prefix}_condition_summary.csv",
        "paired_contrasts": output_root / "results" / f"{prefix}_paired_contrasts.csv",
        "repeat_stability": output_root / "results" / f"{prefix}_repeat_stability.csv",
        "li_label_sensitivity": output_root / "results" / f"{prefix}_li_label_sensitivity.csv",
        "partition_index": output_root / "artifacts" / f"{prefix}_partition_index.csv",
        "predictions": output_root / "artifacts" / f"{prefix}_predictions.npz",
        "kernel_diagnostics": output_root / "diagnostics" / f"{prefix}_kernel_diagnostics.csv",
        "kernel_pairwise": output_root / "diagnostics" / f"{prefix}_kernel_pairwise_alignment.csv",
        "fused_diagnostics": output_root / "diagnostics" / f"{prefix}_fused_affinity_diagnostics.csv",
        "gfm_overlap": output_root / "diagnostics" / f"{prefix}_gfm_overlap.csv",
        "gfm_membership_summary": output_root / "diagnostics" / f"{prefix}_gfm_membership_summary.csv",
        "spectrum": output_root / "diagnostics" / f"{prefix}_spectrum.csv",
        "integrity": output_root / "diagnostics" / f"{prefix}_integrity_report.json",
        "manifest": output_root / "artifacts" / f"{prefix}_analysis_manifest.json",
    }
    required_paths = [path for key, path in paths.items() if key != "li_label_sensitivity"]
    if args.dataset == "li":
        required_paths.append(paths["li_label_sensitivity"])
    existing = [str(path) for path in required_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite campaign outputs: {existing}")

    atomic_csv(raw, paths["raw_metrics"])
    atomic_csv(seed_means, paths["seed_means"])
    atomic_csv(summary, paths["condition_summary"])
    atomic_csv(pd.DataFrame(per_task_contrasts), paths["paired_contrasts"])
    atomic_csv(pd.DataFrame(stability_rows), paths["repeat_stability"])
    if args.dataset == "li":
        atomic_csv(li_sensitivity, paths["li_label_sensitivity"])
    atomic_csv(raw.drop(columns=list(METRICS)), paths["partition_index"])
    atomic_npz(paths["predictions"], labels=partition_array)
    atomic_csv(pd.DataFrame(kernel_rows), paths["kernel_diagnostics"])
    atomic_csv(pd.DataFrame(pairwise_rows), paths["kernel_pairwise"])
    atomic_csv(pd.DataFrame(fused_rows), paths["fused_diagnostics"])
    atomic_csv(gfm_overlap, paths["gfm_overlap"])
    atomic_csv(gfm_membership_summary, paths["gfm_membership_summary"])
    atomic_csv(pd.DataFrame(spectrum_rows), paths["spectrum"])

    integrity = {
        "schema_version": 1,
        "status": "complete",
        "dataset": args.dataset,
        "regime": args.regime,
        "expected_seeds": args.seeds,
        "observed_seed_count": int(raw["seed"].nunique()),
        "expected_repeats": args.repeats,
        "expected_n_gfm": args.n_gfm,
        "expected_rows": expected_rows,
        "observed_rows": len(raw),
        "n_cells": n_cells,
        "cell_ids_sha256": inputs["cell_hash"],
        "conditions": sorted(raw["condition"].unique()),
        "preserve_diagonal": True,
        "cross_gfm_normalization": "off_diagonal_frobenius",
    }
    atomic_json(integrity, paths["integrity"])
    manifest = {
        **integrity,
        "campaign_id": args.campaign_id,
        "campaign_instance_id": args.campaign_instance_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "k_train": args.k_train,
        "k_cluster": args.k_cluster,
        "alpha": args.alpha,
        "beta": args.beta,
        "lambdas": {
            "lambda_config_id": args.lambda_config_id,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "lambda3": args.lambda3,
        },
        "inputs": {
            "kernels_dir": str(args.kernels_dir),
            "latents": str(args.latents),
            "reference_predictions": str(args.reference_predictions),
            "reference_metrics": str(args.reference_metrics),
            "cell_metadata": str(args.cell_metadata),
            "gfm_membership": str(args.gfm_membership),
        },
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    atomic_json(manifest, paths["manifest"])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-instance-id", required=True)
    parser.add_argument("--dataset", required=True, choices=("darmanis", "li", "manno", "muraro"))
    parser.add_argument("--regime", required=True, choices=("oracle", "historical"))
    parser.add_argument("--regime-role", required=True, choices=("primary", "sensitivity"))
    parser.add_argument("--k-train", required=True, type=int)
    parser.add_argument("--k-cluster", required=True, type=int)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--beta", required=True, type=float)
    parser.add_argument("--lambda-config-id", required=True)
    parser.add_argument("--lambda1", required=True, type=float)
    parser.add_argument("--lambda2", required=True, type=float)
    parser.add_argument("--lambda3", required=True, type=float)
    parser.add_argument("--n-gfm", required=True, type=int)
    parser.add_argument("--projection-dimension", required=True, type=int)
    parser.add_argument("--seeds", required=True, type=parse_int_list)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--kmeans-times", required=True, type=int)
    parser.add_argument("--n-neighbour", required=True, type=int)
    parser.add_argument("--red-global", required=True)
    parser.add_argument("--red-local", required=True)
    parser.add_argument("--thread-num", default=16, type=int)
    parser.add_argument("--kernels-dir", required=True, type=Path)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--reference-predictions", required=True, type=Path)
    parser.add_argument("--reference-metrics", required=True, type=Path)
    parser.add_argument("--cell-metadata", required=True, type=Path)
    parser.add_argument("--gfm-membership", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if args.k_train != args.k_cluster:
        parser.error("Campaign 3 requires k_train == k_cluster.")
    if args.repeats < 1 or args.n_gfm < 1 or args.thread_num < 1:
        parser.error("repeats, n_gfm and thread-num must be positive.")
    if args.alpha < 0 or args.beta < 0 or args.alpha + args.beta <= 0:
        parser.error("alpha/beta must be non-negative and not both zero.")
    return args


def main() -> None:
    paths = analyze(parse_args())
    for name, path in paths.items():
        if path.exists():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
