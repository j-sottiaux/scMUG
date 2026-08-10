#!/usr/bin/env python3
"""Validate and consolidate one complete DMKCN k-fusion campaign-3 instance."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASETS = ("darmanis", "li", "manno", "muraro")
REGIMES = ("oracle", "historical")
METRICS = ("NMI", "ARI", "ACC")
PRIMARY = "direct_mean_frobenius"
REFERENCE = "cd_all_gfm"


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


def fdr_bh(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    result = np.full(len(array), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(array))
    if not len(finite):
        return pd.Series(result, index=values.index)
    order = finite[np.argsort(array[finite])]
    ranked = array[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return pd.Series(result, index=values.index)


def exact_sign_flip_test(differences: np.ndarray) -> float:
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff)]
    if not len(diff):
        return np.nan
    observed = abs(float(np.mean(diff)))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(diff)):
        statistics.append(abs(float(np.mean(diff * np.asarray(signs)))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def parse_elapsed_value(value: str) -> float:
    fields = value.strip().split(":")
    numbers = [float(field) for field in fields]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
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
                fields["max_rss_mb_task"] = float(line.rsplit(":", 1)[1]) / 1024.0
            elif line.startswith("Exit status:"):
                fields["exit_status"] = float(line.rsplit(":", 1)[1])
    required = {
        "user_cpu_seconds",
        "system_cpu_seconds",
        "cpu_percent",
        "task_wall_seconds",
        "max_rss_mb_task",
        "exit_status",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"{path}: missing resource fields {missing}.")
    fields["total_cpu_seconds"] = (
        fields["user_cpu_seconds"] + fields["system_cpu_seconds"]
    )
    return fields


def paired_effect(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    dataset: str,
    regime: str,
    metric: str,
    contrast: str,
    family: str,
) -> dict[str, Any]:
    left_values = left.set_index("seed")[metric]
    right_values = right.set_index("seed")[metric]
    common = sorted(set(left_values.index) & set(right_values.index))
    if len(common) != 10:
        raise ValueError(
            f"{dataset}/{regime}/{contrast}/{metric}: expected 10 paired seeds, "
            f"got {len(common)}."
        )
    diff = left_values.loc[common].to_numpy(dtype=float) - right_values.loc[
        common
    ].to_numpy(dtype=float)
    return {
        "family": family,
        "dataset": dataset,
        "regime": regime,
        "contrast": contrast,
        "direction": "left_minus_right",
        "metric": metric,
        "n_pairs": len(common),
        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "std_difference": float(np.std(diff, ddof=1)),
        "wins_left": int(np.sum(diff > 0)),
        "wins_right": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
        "p_value_exact_sign_flip": exact_sign_flip_test(diff),
        "paired_differences": json.dumps([float(value) for value in diff]),
    }


def _condition(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    selected = frame[frame["condition"] == condition].copy()
    if selected.empty:
        raise ValueError(f"Missing condition: {condition}")
    if condition in {"direct_single_gfm", "cd_single_gfm"}:
        selected = selected.groupby("seed", as_index=False)[list(METRICS)].mean()
    return selected


def build_contrasts(seed_means: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset in DATASETS:
        for regime in REGIMES:
            frame = seed_means[
                (seed_means["dataset"] == dataset)
                & (seed_means["regime"] == regime)
            ]
            if frame.empty:
                raise ValueError(f"Missing seed means for {dataset}/{regime}.")
            primary = _condition(frame, PRIMARY)
            reference = _condition(frame, REFERENCE)
            raw_mean = _condition(frame, "direct_mean_raw")
            direct_single = _condition(frame, "direct_single_gfm")
            cd_single = _condition(frame, "cd_single_gfm")
            for metric in METRICS:
                if regime == "oracle" and metric == "NMI":
                    family = "primary_nmi_oracle"
                elif regime == "oracle" and metric in {"ARI", "ACC"}:
                    family = "secondary_ari_acc_oracle"
                else:
                    family = "exploratory"
                records.append(
                    paired_effect(
                        primary,
                        reference,
                        dataset=dataset,
                        regime=regime,
                        metric=metric,
                        contrast="direct_mean_frobenius_minus_cd_all_gfm",
                        family=family,
                    )
                )
                records.append(
                    paired_effect(
                        primary,
                        raw_mean,
                        dataset=dataset,
                        regime=regime,
                        metric=metric,
                        contrast="frobenius_mean_minus_raw_mean",
                        family="exploratory",
                    )
                )
                records.append(
                    paired_effect(
                        primary,
                        direct_single,
                        dataset=dataset,
                        regime=regime,
                        metric=metric,
                        contrast="fused_direct_minus_mean_single_direct",
                        family="exploratory",
                    )
                )

                primary_index = primary.set_index("seed")[metric]
                reference_index = reference.set_index("seed")[metric]
                direct_index = direct_single.set_index("seed")[metric]
                cd_index = cd_single.set_index("seed")[metric]
                common = sorted(
                    set(primary_index.index)
                    & set(reference_index.index)
                    & set(direct_index.index)
                    & set(cd_index.index)
                )
                interaction = pd.DataFrame(
                    {
                        "seed": common,
                        metric: (
                            primary_index.loc[common].to_numpy()
                            - direct_index.loc[common].to_numpy()
                            - reference_index.loc[common].to_numpy()
                            + cd_index.loc[common].to_numpy()
                        ),
                    }
                )
                zero = pd.DataFrame({"seed": common, metric: np.zeros(len(common))})
                records.append(
                    paired_effect(
                        interaction,
                        zero,
                        dataset=dataset,
                        regime=regime,
                        metric=metric,
                        contrast="descriptive_multiview_by_downstream_interaction",
                        family="exploratory",
                    )
                )
    contrasts = pd.DataFrame(records)
    contrasts["p_value_fdr_bh"] = np.nan
    for family in ("primary_nmi_oracle", "secondary_ari_acc_oracle"):
        mask = contrasts["family"] == family
        contrasts.loc[mask, "p_value_fdr_bh"] = fdr_bh(
            contrasts.loc[mask, "p_value_exact_sign_flip"]
        )
    contrasts["significant_fdr_0_05"] = (
        contrasts["p_value_fdr_bh"].notna()
        & (contrasts["p_value_fdr_bh"] < 0.05)
    )
    return contrasts


def load_complete_instance(input_root: Path, instance_id: str):
    manifests = sorted(
        input_root.glob("analysis/*/*/artifacts/*_analysis_manifest.json")
    )
    inventory = []
    issues = []
    by_task: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest_path in manifests:
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            key = (manifest["dataset"], manifest["regime"])
            inventory.append(
                {
                    "dataset": key[0],
                    "regime": key[1],
                    "status": manifest.get("status"),
                    "manifest": str(manifest_path),
                }
            )
            if manifest.get("campaign_instance_id") != instance_id:
                issues.append({"issue": "instance_mismatch", "path": str(manifest_path)})
                continue
            if key in by_task:
                issues.append({"issue": "duplicate_task", "path": str(manifest_path)})
                continue
            if manifest.get("status") != "complete":
                issues.append({"issue": "incomplete_manifest", "path": str(manifest_path)})
                continue
            by_task[key] = manifest
        except Exception as exc:
            issues.append(
                {
                    "issue": f"unparsable_manifest:{type(exc).__name__}:{exc}",
                    "path": str(manifest_path),
                }
            )

    expected = {(dataset, regime) for dataset in DATASETS for regime in REGIMES}
    for missing in sorted(expected - set(by_task)):
        issues.append({"issue": "missing_task", "path": f"{missing[0]}/{missing[1]}"})
    for extra in sorted(set(by_task) - expected):
        issues.append({"issue": "unexpected_task", "path": f"{extra[0]}/{extra[1]}"})
    return by_task, pd.DataFrame(inventory), pd.DataFrame(issues)


def load_computation_metrics(
    input_root: Path,
    tasks: dict[tuple[str, str], dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resource_rows = []
    timing_frames = []
    for dataset, regime in sorted(tasks):
        training_manifests = list(
            input_root.glob(
                f"training/{dataset}/{regime}/artifacts/*_training_manifest.json"
            )
        )
        if len(training_manifests) != 1:
            raise ValueError(
                f"{dataset}/{regime}: expected one training manifest, got "
                f"{len(training_manifests)}."
            )
        with training_manifests[0].open("r", encoding="utf-8") as handle:
            training = json.load(handle)
        if training.get("status") != "completed" or training.get("exit_status") != 0:
            raise ValueError(f"Incomplete training manifest: {training_manifests[0]}")
        training_resource = Path(training["paths"]["resource_usage"])
        resource_rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "phase": "training",
                "source_file": str(training_resource),
                **parse_resource_usage(training_resource),
            }
        )
        timing_path = Path(training["paths"]["computation_outfile"])
        timing = pd.read_csv(timing_path)
        required_timing = {
            "stage",
            "elapsed_seconds",
            "peak_gpu_allocated_mb",
            "peak_gpu_reserved_mb",
            "peak_rss_mb",
        }
        missing = sorted(required_timing.difference(timing.columns))
        if missing:
            raise ValueError(f"{timing_path}: missing timing columns {missing}.")
        timing.insert(0, "regime", regime)
        timing.insert(0, "dataset", dataset)
        timing["source_file"] = str(timing_path)
        timing_frames.append(timing)

        analysis_resources = list(
            input_root.glob(
                f"analysis/{dataset}/{regime}/computation/*_resource_usage.txt"
            )
        )
        if len(analysis_resources) != 1:
            raise ValueError(
                f"{dataset}/{regime}: expected one analysis resource file, got "
                f"{len(analysis_resources)}."
            )
        resource_rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "phase": "analysis",
                "source_file": str(analysis_resources[0]),
                **parse_resource_usage(analysis_resources[0]),
            }
        )

    resources = pd.DataFrame(resource_rows)
    timings = pd.concat(timing_frames, ignore_index=True)
    phase_summary = (
        resources.groupby("phase", as_index=False)
        .agg(
            total_task_wall_seconds=("task_wall_seconds", "sum"),
            maximum_task_wall_seconds=("task_wall_seconds", "max"),
            total_cpu_seconds=("total_cpu_seconds", "sum"),
            maximum_rss_mb=("max_rss_mb_task", "max"),
            mean_cpu_percent=("cpu_percent", "mean"),
            n_tasks=("phase", "size"),
        )
    )
    training_summary = {
        "phase": "training_internal_recorder",
        "total_task_wall_seconds": float(
            timings.loc[timings["stage"] == "pipeline_wall_total", "elapsed_seconds"].sum()
        ),
        "maximum_task_wall_seconds": float(
            timings.loc[timings["stage"] == "pipeline_wall_total", "elapsed_seconds"].max()
        ),
        "total_cpu_seconds": np.nan,
        "maximum_rss_mb": float(timings["peak_rss_mb"].max()),
        "mean_cpu_percent": np.nan,
        "n_tasks": int(
            timings.loc[timings["stage"] == "pipeline_wall_total", ["dataset", "regime"]]
            .drop_duplicates()
            .shape[0]
        ),
        "maximum_gpu_allocated_mb": float(timings["peak_gpu_allocated_mb"].max()),
        "maximum_gpu_reserved_mb": float(timings["peak_gpu_reserved_mb"].max()),
    }
    phase_summary["maximum_gpu_allocated_mb"] = np.nan
    phase_summary["maximum_gpu_reserved_mb"] = np.nan
    phase_summary = pd.concat(
        [phase_summary, pd.DataFrame([training_summary])], ignore_index=True
    )
    return resources, timings, phase_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--campaign-instance-id", required=True)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    tasks, inventory, issues = load_complete_instance(
        args.input_root, args.campaign_instance_id
    )
    if not issues.empty:
        args.outdir.mkdir(parents=True, exist_ok=True)
        issue_path = args.outdir / "integrity_issues.csv"
        if not issue_path.exists():
            atomic_csv(issues, issue_path)
        raise RuntimeError(
            f"Campaign instance is incomplete or ambiguous; see {issue_path}."
        )

    raw_frames = []
    seed_frames = []
    summary_frames = []
    kernel_frames = []
    pairwise_frames = []
    fused_frames = []
    overlap_frames = []
    membership_summary_frames = []
    stability_frames = []
    spectrum_frames = []
    li_frames = []
    for key in sorted(tasks):
        outputs = tasks[key]["outputs"]
        required = {
            "raw_metrics",
            "seed_means",
            "condition_summary",
            "kernel_diagnostics",
            "kernel_pairwise",
            "fused_diagnostics",
            "gfm_overlap",
            "gfm_membership_summary",
            "repeat_stability",
            "spectrum",
        }
        missing_outputs = sorted(required.difference(outputs))
        if missing_outputs:
            raise ValueError(f"{key}: manifest misses outputs {missing_outputs}.")
        frames = {
            name: pd.read_csv(outputs[name]) for name in required
        }
        raw_frames.append(frames["raw_metrics"])
        seed_frames.append(frames["seed_means"])
        summary_frames.append(frames["condition_summary"])
        kernel_frames.append(frames["kernel_diagnostics"])
        pairwise_frames.append(frames["kernel_pairwise"])
        fused_frames.append(frames["fused_diagnostics"])
        overlap_frames.append(frames["gfm_overlap"])
        membership_summary_frames.append(frames["gfm_membership_summary"])
        stability_frames.append(frames["repeat_stability"])
        spectrum_frames.append(frames["spectrum"])
        li_path = outputs.get("li_label_sensitivity")
        if key[0] == "li" and li_path and Path(li_path).is_file():
            li_frames.append(pd.read_csv(li_path))

    raw = pd.concat(raw_frames, ignore_index=True)
    seed_means = pd.concat(seed_frames, ignore_index=True)
    if len(raw) != 2640:
        raise ValueError(f"Expected 2640 raw metric rows, got {len(raw)}.")
    duplicate_key = ["dataset", "regime", "condition", "gfm_index", "seed", "repeat"]
    if raw.duplicated(duplicate_key).any():
        raise ValueError("Duplicate raw metric combinations detected.")
    contrasts = build_contrasts(seed_means)
    computation_resources, computation_timings, computation_summary = (
        load_computation_metrics(args.input_root, tasks)
    )

    outputs = {
        "campaign_inventory.csv": inventory,
        "integrity_issues.csv": pd.DataFrame(columns=["issue", "path"]),
        "all_raw_metrics.csv": raw,
        "all_seed_means.csv": seed_means,
        "condition_summary.csv": pd.concat(summary_frames, ignore_index=True),
        "primary_contrasts.csv": contrasts[contrasts["family"] == "primary_nmi_oracle"],
        "secondary_contrasts.csv": contrasts[
            contrasts["family"] == "secondary_ari_acc_oracle"
        ],
        "exploratory_contrasts.csv": contrasts[contrasts["family"] == "exploratory"],
        "statistical_tests.csv": contrasts,
        "kernel_diagnostics.csv": pd.concat(kernel_frames, ignore_index=True),
        "kernel_pairwise_alignment.csv": pd.concat(pairwise_frames, ignore_index=True),
        "fused_affinity_diagnostics.csv": pd.concat(fused_frames, ignore_index=True),
        "gfm_overlap.csv": pd.concat(overlap_frames, ignore_index=True),
        "gfm_membership_summary.csv": pd.concat(
            membership_summary_frames, ignore_index=True
        ),
        "repeat_stability.csv": pd.concat(stability_frames, ignore_index=True),
        "spectrum.csv": pd.concat(spectrum_frames, ignore_index=True),
        "computation_resources.csv": computation_resources,
        "computation_timings.csv": computation_timings,
        "computation_summary.csv": computation_summary,
    }
    if li_frames:
        outputs["li_label_sensitivity.csv"] = pd.concat(li_frames, ignore_index=True)
    existing = [str(args.outdir / name) for name in outputs if (args.outdir / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite consolidated outputs: {existing}")
    for name, frame in outputs.items():
        atomic_csv(frame, args.outdir / name)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": "dmkcn_kfusion_campaign3",
        "campaign_instance_id": args.campaign_instance_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_tasks": len(tasks),
        "n_raw_metric_rows": len(raw),
        "n_seed_mean_rows": len(seed_means),
        "primary_metric": "NMI",
        "primary_regime": "oracle",
        "primary_contrast": "direct_mean_frobenius_minus_cd_all_gfm",
        "multiple_testing": {
            "primary": "BH across four dataset-level NMI tests",
            "secondary": "BH across eight dataset-level ARI/ACC tests",
            "exploratory": "effect sizes only; no significance label",
        },
        "outputs": {name: str(args.outdir / name) for name in outputs},
    }
    atomic_json(summary, args.outdir / "campaign_summary.json")
    print(f"Campaign 3 consolidation complete: {args.outdir}")


if __name__ == "__main__":
    main()
