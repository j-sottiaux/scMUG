"""Wall-time and process-memory instrumentation for scMUG experiments."""

from __future__ import annotations

import atexit
import csv
import fcntl
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import torch


FIELDS = [
    "created_at_utc",
    "experiment_id",
    "run_id",
    "dataset",
    "k",
    "pipeline",
    "seed",
    "repeat",
    "gfm_index",
    "stage",
    "alpha",
    "beta",
    "elapsed_seconds",
    "n_cells",
    "n_genes",
    "n_hvg",
    "n_gfm",
    "gfm_gene_count",
    "device",
    "peak_gpu_allocated_mb",
    "peak_gpu_reserved_mb",
    "peak_rss_mb",
    "slurm_job_id",
    "slurm_array_task_id",
    "status",
]


def synchronize_cuda() -> None:
    """Wait for pending CUDA work so wall-clock timings are not truncated."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def synchronized_start() -> float:
    synchronize_cuda()
    return time.perf_counter()


def synchronized_elapsed(started: float) -> float:
    synchronize_cuda()
    return time.perf_counter() - started


def peak_rss_mb() -> float:
    """Return process peak RSS, normalised across Linux and macOS."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def gpu_peaks_mb() -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    scale = 1024.0 * 1024.0
    return (
        float(torch.cuda.max_memory_allocated()) / scale,
        float(torch.cuda.max_memory_reserved()) / scale,
    )


class ComputationRecorder:
    """Append computation records to one run-specific CSV file."""

    def __init__(
        self,
        output_path: str | Path | None,
        *,
        experiment_id: str,
        run_id: str,
        dataset: str,
        k: int,
        pipeline: str,
        n_gfm: int,
    ) -> None:
        self.output_path = Path(output_path) if output_path else None
        self.identity = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "dataset": dataset,
            "k": int(k),
            "pipeline": pipeline,
            "n_gfm": int(n_gfm),
        }
        self.dimensions: dict[str, int | None] = {
            "n_cells": None,
            "n_genes": None,
            "n_hvg": None,
        }
        self.rows: list[dict[str, Any]] = []
        self._flushed_count = 0
        atexit.register(self.flush)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def set_dimensions(self, *, n_cells: int, n_genes: int, n_hvg: int) -> None:
        self.dimensions = {
            "n_cells": int(n_cells),
            "n_genes": int(n_genes),
            "n_hvg": int(n_hvg),
        }

    def start(self) -> float:
        return synchronized_start()

    def finish(
        self,
        started: float,
        stage: str,
        **metadata: Any,
    ) -> float:
        elapsed = synchronized_elapsed(started)
        self.record(stage, elapsed, **metadata)
        return elapsed

    def record(
        self,
        stage: str,
        elapsed_seconds: float,
        *,
        seed: int | None = None,
        repeat: int | None = None,
        gfm_index: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gfm_gene_count: int | None = None,
        status: str = "ok",
    ) -> None:
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        allocated_mb, reserved_mb = gpu_peaks_mb()
        row = {
            **self.identity,
            **self.dimensions,
            "created_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "seed": seed,
            "repeat": repeat,
            "gfm_index": gfm_index,
            "stage": stage,
            "alpha": alpha,
            "beta": beta,
            "elapsed_seconds": float(elapsed_seconds),
            "gfm_gene_count": gfm_gene_count,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "peak_gpu_allocated_mb": allocated_mb,
            "peak_gpu_reserved_mb": reserved_mb,
            "peak_rss_mb": peak_rss_mb(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "status": status,
        }
        self.rows.append(row)

    def elapsed_sum(
        self,
        stages: set[str],
        *,
        seed: int | None = None,
    ) -> float:
        return float(
            sum(
                row["elapsed_seconds"]
                for row in self.rows
                if row["stage"] in stages
                and (seed is None or row["seed"] == seed)
            )
        )

    def flush(self) -> None:
        if self.output_path is None or self._flushed_count >= len(self.rows):
            return
        pending = self.rows[self._flushed_count :]
        self._append_many(pending)
        self._flushed_count = len(self.rows)

    def _append_many(self, rows: list[dict[str, Any]]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a+", newline="", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0, os.SEEK_END)
            write_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
