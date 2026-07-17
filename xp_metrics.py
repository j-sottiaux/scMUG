"""Build scMUG-style metric summaries with structured provenance logs.

This script summarizes the full scMUG output files and, optionally, the C/D
ablation table. It collapses repeated final clustering runs to one mean value per
seed, then reports median, mean, standard deviation, min and max across seeds.

The default output keeps only the canonical full-pipeline alpha/beta pair for the
full AE and DMKCN runs. Ablation rows are always retained because their alpha/beta
pairs define the ablated condition itself.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRICS = ["NMI", "ARI", "ACC"]
FULL_LINE_RE = re.compile(
    r"dbname:(?P<dataset>\S+)\s+"
    r"round:(?P<seed>\S+)\s+"
    r"alpha:(?P<alpha>\S+)\s+"
    r"beta:(?P<beta>\S+)\s+"
    r"acc:\s*(?P<ACC>[\d.eE+-]+)\s+"
    r"ari:\s*(?P<ARI>[\d.eE+-]+)\s+"
    r"nmi:\s*(?P<NMI>[\d.eE+-]+)"
)


def _float_or_nan(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _same_float(a: Any, b: Any, tol: float = 1e-12) -> bool:
    a_float = _float_or_nan(a)
    b_float = _float_or_nan(b)
    if np.isnan(a_float) or np.isnan(b_float):
        return False
    return abs(a_float - b_float) <= tol


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_full_txt(
    path: Path,
    dataset: str,
    model: str,
    canonical_alpha: float,
    canonical_beta: float,
    include_all_alpha_beta: bool,
) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = FULL_LINE_RE.search(line)
            if match is None:
                continue
            parsed = match.groupdict()
            rows.append(
                {
                    "dataset": dataset or parsed["dataset"],
                    "model": model,
                    "condition": "full",
                    "seed": int(float(parsed["seed"])),
                    "repeat": pd.NA,
                    "alpha": float(parsed["alpha"]),
                    "beta": float(parsed["beta"]),
                    "c_mode": "full",
                    "d_mode": "spectral_precomputed",
                    "NMI": float(parsed["NMI"]),
                    "ARI": float(parsed["ARI"]),
                    "ACC": float(parsed["ACC"]),
                    "source": "full_run",
                    "source_file": str(path),
                }
            )

    if not rows:
        raise ValueError(f"No valid metric rows parsed from {path}.")

    frame = pd.DataFrame(rows)
    frame["repeat"] = (
        frame.groupby(
            ["dataset", "model", "condition", "seed", "alpha", "beta"],
            dropna=False,
        ).cumcount()
    )

    if include_all_alpha_beta:
        frame["canonical_alpha_beta"] = frame.apply(
            lambda row: _same_float(row["alpha"], canonical_alpha)
            and _same_float(row["beta"], canonical_beta),
            axis=1,
        )
        return frame

    mask = frame.apply(
        lambda row: _same_float(row["alpha"], canonical_alpha)
        and _same_float(row["beta"], canonical_beta),
        axis=1,
    )
    filtered = frame.loc[mask].copy()
    if filtered.empty:
        available = sorted(set(zip(frame["alpha"], frame["beta"])))
        raise ValueError(
            f"{path}: no rows with canonical alpha={canonical_alpha}, "
            f"beta={canonical_beta}. Available pairs: {available}"
        )
    filtered["canonical_alpha_beta"] = True
    return filtered


def read_ablation_tsv(path: Path, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"arm", "seed", "repeat", "method", "acc", "ari", "nmi"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    condition_map = {
        "no_C": "without_C",
        "no_D": "without_D",
        "no_CD_direct_spectral": "without_CD_direct_spectral",
        "C_full_D_spectral": "full_recomputed_from_latents",
        "C_global_only_D_spectral": "without_C_local",
        "C_local_only_D_spectral": "without_C_global",
        "no_C_D_spectral_direct_knn": "without_C",
        "no_C_D_spectral_C_input_knn": "without_C",
        "C_full_no_D_kmeans_rows": "without_D",
        "no_C_no_D_kmeans_concat": "without_CD",
        "no_C_no_D_kmeans_C_input": "without_CD",
    }
    model_map = {
        "autoencoder": "scMUG",
        "dmkcn": "scMUG-DMKCN",
    }

    out = pd.DataFrame(
        {
            "dataset": dataset,
            "model": frame["arm"].replace(model_map),
            "condition": frame["method"].replace(condition_map),
            "seed": frame["seed"].astype(int),
            "repeat": frame["repeat"],
            "alpha": frame["alpha"] if "alpha" in frame.columns else pd.NA,
            "beta": frame["beta"] if "beta" in frame.columns else pd.NA,
            "c_mode": frame["c_mode"] if "c_mode" in frame.columns else pd.NA,
            "d_mode": frame["d_mode"] if "d_mode" in frame.columns else pd.NA,
            "NMI": frame["nmi"].astype(float),
            "ARI": frame["ari"].astype(float),
            "ACC": frame["acc"].astype(float),
            "source": "ablation",
            "source_file": str(path),
            "canonical_alpha_beta": pd.NA,
        }
    )
    return out


def metric_stats(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)[METRICS]
    summary = grouped.agg(["median", "mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    count = grouped.size().reset_index(name="n_seeds")
    return summary.merge(count, on=group_cols, how="left")


def summarize(raw_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = [
        "experiment_id",
        "run_id",
        "dataset",
        "k",
        "model",
        "condition",
        "alpha",
        "beta",
        "c_mode",
        "d_mode",
        "seed",
    ]
    summary_cols = key_cols[:-1]

    seed_means = (
        raw_metrics.groupby(key_cols, dropna=False)[METRICS]
        .mean()
        .reset_index()
        .sort_values(key_cols)
    )
    condition_summary = metric_stats(seed_means, summary_cols).sort_values(summary_cols)
    return seed_means, condition_summary


def build_run_details(
    raw_metrics: pd.DataFrame,
    args: argparse.Namespace,
    output_paths: dict[str, Path],
) -> pd.DataFrame:
    rows = []
    for (source_file, model, source), frame in raw_metrics.groupby(
        ["source_file", "model", "source"],
        dropna=False,
    ):
        pairs = list(
            {
                (None if pd.isna(alpha) else float(alpha), None if pd.isna(beta) else float(beta))
                for alpha, beta in zip(frame["alpha"], frame["beta"])
            }
        )
        pairs = sorted(pairs, key=lambda pair: (str(pair[0]), str(pair[1])))
        rows.append(
            {
                "experiment_id": args.experiment_id,
                "run_id": args.run_id,
                "dataset": args.dataset,
                "k": args.cluster_number,
                "source_file": source_file,
                "model": model,
                "source": source,
                "conditions": ",".join(sorted(map(str, frame["condition"].unique()))),
                "n_raw_rows": int(len(frame)),
                "n_seeds": int(frame["seed"].nunique()),
                "seeds": ",".join(str(x) for x in sorted(frame["seed"].unique())),
                "n_repeats_max": int(frame.groupby("seed").size().max()),
                "alpha_beta_pairs": json.dumps(pairs),
                "canonical_alpha": args.full_alpha,
                "canonical_beta": args.full_beta,
                "include_all_alpha_beta": bool(args.include_all_alpha_beta),
                "raw_metrics": str(output_paths["raw_metrics"]),
                "seed_means": str(output_paths["seed_means"]),
                "condition_summary": str(output_paths["condition_summary"]),
                "created_at": output_paths["created_at"],
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    raw_metrics: pd.DataFrame,
    seed_means: pd.DataFrame,
    condition_summary: pd.DataFrame,
    run_details: pd.DataFrame,
    output_paths: dict[str, Path | str],
    provenance: dict[str, Any],
) -> None:
    output_paths["outdir"].mkdir(parents=True, exist_ok=True)
    output_paths["metadata_outdir"].mkdir(parents=True, exist_ok=True)
    raw_metrics.to_csv(output_paths["raw_metrics"], index=False)
    seed_means.to_csv(output_paths["seed_means"], index=False)
    condition_summary.to_csv(output_paths["condition_summary"], index=False)
    run_details.to_csv(output_paths["run_details"], index=False)

    with output_paths["run_started"].open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, sort_keys=True, indent=2)
        handle.write("\n")
    append_jsonl(output_paths["provenance"], provenance)


def make_output_paths(
    outdir: Path,
    metadata_outdir: Path,
    output_prefix: str,
) -> dict[str, Path | str]:
    if output_prefix != Path(output_prefix).name or output_prefix in {".", ".."}:
        raise ValueError("output_prefix must be a filename prefix, not a path")
    created_at = time.strftime("%Y%m%d_%H%M%S")
    return {
        "outdir": outdir,
        "metadata_outdir": metadata_outdir,
        "created_at": created_at,
        "raw_metrics": outdir / f"{output_prefix}_raw_metrics.csv",
        "seed_means": outdir / f"{output_prefix}_seed_means.csv",
        "condition_summary": outdir / f"{output_prefix}_condition_summary.csv",
        "run_details": outdir / f"{output_prefix}_run_details.csv",
        "provenance": metadata_outdir / f"{output_prefix}_metrics_events.jsonl",
        "run_started": metadata_outdir / f"{output_prefix}_metrics_manifest.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cluster-number", required=True, type=int)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full-autoencoder", required=True, type=Path)
    parser.add_argument("--full-dmkcn", required=True, type=Path)
    parser.add_argument("--ablation", type=Path, default=None)
    parser.add_argument("--full-alpha", required=True, type=float)
    parser.add_argument("--full-beta", required=True, type=float)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--metadata-outdir", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--include-all-alpha-beta",
        action="store_true",
        help="Keep all full-pipeline alpha/beta pairs instead of only the canonical pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = make_output_paths(
        args.outdir,
        args.metadata_outdir,
        args.output_prefix,
    )

    frames = [
        parse_full_txt(
            args.full_autoencoder,
            dataset=args.dataset,
            model="scMUG",
            canonical_alpha=args.full_alpha,
            canonical_beta=args.full_beta,
            include_all_alpha_beta=args.include_all_alpha_beta,
        ),
        parse_full_txt(
            args.full_dmkcn,
            dataset=args.dataset,
            model="scMUG-DMKCN",
            canonical_alpha=args.full_alpha,
            canonical_beta=args.full_beta,
            include_all_alpha_beta=args.include_all_alpha_beta,
        ),
    ]
    if args.ablation is not None:
        frames.append(read_ablation_tsv(args.ablation, dataset=args.dataset))

    raw_metrics = pd.concat(frames, ignore_index=True)
    raw_metrics.insert(0, "experiment_id", args.experiment_id)
    raw_metrics.insert(1, "run_id", args.run_id)
    raw_metrics.insert(3, "k", int(args.cluster_number))
    seed_means, condition_summary = summarize(raw_metrics)

    provenance = {
        "created_at": output_paths["created_at"],
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "dataset": args.dataset,
        "k": args.cluster_number,
        "inputs": {
            "full_autoencoder": str(args.full_autoencoder),
            "full_dmkcn": str(args.full_dmkcn),
            "ablation": None if args.ablation is None else str(args.ablation),
        },
        "outputs": {
            key: str(value)
            for key, value in output_paths.items()
            if key not in {"outdir", "metadata_outdir", "created_at"}
        },
        "parameters": {
            "full_alpha": args.full_alpha,
            "full_beta": args.full_beta,
            "include_all_alpha_beta": args.include_all_alpha_beta,
        },
        "n_raw_rows": int(len(raw_metrics)),
        "n_seed_mean_rows": int(len(seed_means)),
        "n_summary_rows": int(len(condition_summary)),
        "seeds": [int(x) for x in sorted(raw_metrics["seed"].unique())],
    }
    run_details = build_run_details(raw_metrics, args, output_paths)

    write_outputs(
        raw_metrics,
        seed_means,
        condition_summary,
        run_details,
        output_paths,
        provenance,
    )

    print(f"Loaded {len(raw_metrics)} raw metric rows")
    print(f"Saved raw metrics to {output_paths['raw_metrics']}")
    print(f"Saved seed means to {output_paths['seed_means']}")
    print(f"Saved condition summary to {output_paths['condition_summary']}")
    print(f"Saved run details to {output_paths['run_details']}")
    print(f"Saved provenance to {output_paths['provenance']}")


if __name__ == "__main__":
    main()
