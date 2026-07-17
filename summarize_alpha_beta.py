"""Validate the alpha/beta grid and select one pair per dataset and pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINES = ("scMUG", "scMUG-DMKCN")
ARM_TO_PIPELINE = {"autoencoder": "scMUG", "dmkcn": "scMUG-DMKCN"}
DEFAULT_GRID = (
    (0.0, 1.0),
    (0.001, 1.0),
    (0.01, 1.0),
    (0.1, 1.0),
    (1.0, 1.0),
    (1.0, 0.1),
    (1.0, 0.01),
    (1.0, 0.001),
    (1.0, 0.0),
)
REQUIRED_COLUMNS = {
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
}


def parse_grid(value: str) -> tuple[tuple[float, float], ...]:
    pairs = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(f"Invalid alpha:beta pair: {item!r}")
        pair = (float(fields[0]), float(fields[1]))
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise argparse.ArgumentTypeError("Alpha/beta grid cannot be empty.")
    return tuple(pairs)


def discover_grid_rows(input_root: Path, experiment_id: str) -> pd.DataFrame:
    paths = sorted(input_root.glob(f"*/results/{experiment_id}_*_grid.tsv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.loc[frame["experiment_id"] == experiment_id].copy()
        if frame.empty:
            continue
        frame["source_file"] = str(path)
        frame["source_mtime_ns"] = path.stat().st_mtime_ns
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No alpha/beta-grid results found for {experiment_id!r} below {input_root}."
        )
    return pd.concat(frames, ignore_index=True)


def select_task_runs(raw: pd.DataFrame, duplicate_policy: str) -> pd.DataFrame:
    raw = raw.copy()
    raw["model"] = raw["arm"].map(ARM_TO_PIPELINE)
    if raw["model"].isna().any():
        unknown = sorted(raw.loc[raw["model"].isna(), "arm"].unique())
        raise ValueError(f"Unknown representation arms in alpha/beta grid: {unknown}")

    task_key = ["dataset", "model", "k"]
    runs = raw[task_key + ["run_id", "source_file", "source_mtime_ns"]].drop_duplicates()
    duplicates = runs.duplicated(task_key, keep=False)
    if duplicates.any():
        if duplicate_policy == "error":
            raise ValueError(
                "Duplicate alpha/beta task runs found. Resolve them or use "
                "--duplicate-policy latest:\n"
                + runs.loc[duplicates].to_string(index=False)
            )
        keep = (
            runs.sort_values(["source_mtime_ns", "run_id"])
            .drop_duplicates(task_key, keep="last")
            .loc[:, task_key + ["run_id"]]
        )
        raw = raw.merge(keep, on=task_key + ["run_id"], how="inner")
    return raw


def validate_grid(
    raw: pd.DataFrame,
    best_k: pd.DataFrame,
    expected_grid: tuple[tuple[float, float], ...],
    expected_seeds: tuple[int, ...],
    expected_repeats: int,
) -> None:
    expected_tasks = set(zip(best_k["dataset"], best_k["model"], best_k["k"]))
    observed_tasks = set(zip(raw["dataset"], raw["model"], raw["k"]))
    if observed_tasks != expected_tasks:
        raise ValueError(
            f"Alpha/beta task coverage mismatch. Missing: "
            f"{sorted(expected_tasks - observed_tasks)}; unexpected: "
            f"{sorted(observed_tasks - expected_tasks)}"
        )

    invalid_conditions = raw.loc[
        (raw["method"] != "C_full_D_spectral")
        | (raw["c_mode"] != "full")
        | (raw["d_mode"] != "spectral_precomputed")
    ]
    if not invalid_conditions.empty:
        raise ValueError("Alpha/beta tuning contains non-spectral or ablation rows.")

    expected_pairs = set(expected_grid)
    for task, frame in raw.groupby(["dataset", "model", "k"]):
        observed_pairs = set(zip(frame["alpha"], frame["beta"]))
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"{task}: alpha/beta coverage mismatch. Missing: "
                f"{sorted(expected_pairs - observed_pairs)}; unexpected: "
                f"{sorted(observed_pairs - expected_pairs)}"
            )

    expected_seed_set = set(expected_seeds)
    for key, frame in raw.groupby(["dataset", "model", "k", "alpha", "beta"]):
        observed_seeds = set(frame["seed"].astype(int))
        if observed_seeds != expected_seed_set:
            raise ValueError(
                f"{key}: seed coverage mismatch. Missing: "
                f"{sorted(expected_seed_set - observed_seeds)}"
            )
        repeat_counts = frame.groupby("seed")["repeat"].nunique()
        if not (repeat_counts == expected_repeats).all():
            raise ValueError(
                f"{key}: expected {expected_repeats} repeats per seed, observed "
                f"{repeat_counts.to_dict()}"
            )


def summarize_grid(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["experiment_id", "run_id", "dataset", "k", "model", "alpha", "beta"]
    seed_means = (
        raw.groupby(keys + ["seed"], dropna=False)[["nmi", "ari", "acc"]]
        .mean()
        .reset_index()
        .rename(columns={"nmi": "NMI", "ari": "ARI", "acc": "ACC"})
    )
    summary = (
        seed_means.groupby(keys, dropna=False)[["NMI", "ARI", "ACC"]]
        .agg(["median", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    counts = seed_means.groupby(keys, dropna=False).size().reset_index(name="n_seeds")
    return seed_means, summary.merge(counts, on=keys, how="left")


def select_best_pairs(
    summary: pd.DataFrame, grid: tuple[tuple[float, float], ...]
) -> pd.DataFrame:
    grid_order = {pair: index for index, pair in enumerate(grid)}
    ranked = summary.copy()
    ranked["grid_order"] = [
        grid_order[(float(alpha), float(beta))]
        for alpha, beta in zip(ranked["alpha"], ranked["beta"])
    ]
    ranked = ranked.sort_values(
        [
            "dataset",
            "model",
            "NMI_median",
            "ARI_median",
            "ACC_median",
            "grid_order",
        ],
        ascending=[True, True, False, False, False, True],
    )
    ranked["nmi_rank"] = ranked.groupby(["dataset", "model"]).cumcount() + 1
    best = ranked.loc[ranked["nmi_rank"] == 1].copy()
    best["selection_metric"] = "NMI_median"
    best["tie_break_rule"] = "ARI_median,ACC_median,grid_order"
    return best.sort_values(["dataset", "model"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--best-k",
        type=Path,
        default=Path("publication/k_sweep/ksweep_best_k_by_nmi.csv"),
    )
    parser.add_argument("--experiment-id", default="alpha_beta_grid")
    parser.add_argument(
        "--alpha-beta-grid",
        type=parse_grid,
        default=DEFAULT_GRID,
    )
    parser.add_argument(
        "--seeds",
        default="1111,2222,3333,4444,5555,6666,7777,8888,9999,10000",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--duplicate-policy", choices=["error", "latest"], default="error"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("publication/alpha_beta")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_seeds = tuple(int(seed) for seed in args.seeds.split(","))
    best_k = pd.read_csv(args.best_k)
    best_k["k"] = pd.to_numeric(best_k["k"], errors="raise").astype(int)
    raw = discover_grid_rows(args.input_root, args.experiment_id)
    for column in ["k", "seed", "repeat", "alpha", "beta", "nmi", "ari", "acc"]:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    raw = select_task_runs(raw, args.duplicate_policy)
    validate_grid(raw, best_k, args.alpha_beta_grid, expected_seeds, args.repeat)
    seed_means, summary = summarize_grid(raw)
    best = select_best_pairs(summary, args.alpha_beta_grid)

    best_metadata = best_k[
        [
            "dataset",
            "model",
            "published_best_k",
            "ground_truth_k",
            "latent_artifact",
            "source_summary_file",
        ]
    ]
    best = best.merge(best_metadata, on=["dataset", "model"], how="left")

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.outdir / "alpha_beta_raw.csv", index=False)
    seed_means.to_csv(args.outdir / "alpha_beta_seed_means.csv", index=False)
    summary.to_csv(args.outdir / "alpha_beta_summary.csv", index=False)
    best.to_csv(args.outdir / "alpha_beta_best_by_nmi.csv", index=False)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment_id": args.experiment_id,
        "input_root": str(args.input_root),
        "best_k": str(args.best_k),
        "alpha_beta_grid": [list(pair) for pair in args.alpha_beta_grid],
        "seeds": list(expected_seeds),
        "repeat": args.repeat,
        "duplicate_policy": args.duplicate_policy,
        "selection_metric": "NMI_median",
        "tie_break_rule": ["ARI_median", "ACC_median", "grid_order"],
        "n_raw_rows": int(len(raw)),
        "n_seed_mean_rows": int(len(seed_means)),
        "n_summary_rows": int(len(summary)),
    }
    with (args.outdir / "alpha_beta_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(best[["dataset", "model", "k", "alpha", "beta", "NMI_median"]].to_string(index=False))
    print(f"\nWrote alpha/beta summaries to {args.outdir}")


if __name__ == "__main__":
    main()
