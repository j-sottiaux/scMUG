"""Run one command and record portable process resource metrics.

The output deliberately uses the small subset of GNU ``time -v`` labels already
consumed by the campaign-2 consolidator.  Measurement itself relies only on the
Python standard library, because GNU ``time`` is not installed on every Zeus
compute node.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time


def _format_elapsed(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:05.2f}"
    return f"{minutes}:{remaining_seconds:05.2f}"


def _max_rss_kbytes(usage: resource.struct_rusage) -> float:
    # Linux reports KiB; macOS reports bytes. Zeus compute nodes use Linux, but
    # normalizing Darwin keeps local tests and dry-runs interpretable.
    value = float(usage.ru_maxrss)
    return value / 1024.0 if sys.platform == "darwin" else value


def run_and_measure(command: list[str], output_path: Path) -> int:
    """Run ``command``, atomically record its resource use, and return its status."""

    if not command:
        raise ValueError("A command is required after '--'.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite resource file: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"Resource-output directory does not exist: {output_path.parent}"
        )

    started = time.monotonic()
    completed = subprocess.run(command, check=False)
    elapsed_seconds = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    user_seconds = float(usage.ru_utime)
    system_seconds = float(usage.ru_stime)
    cpu_percent = (
        100.0 * (user_seconds + system_seconds) / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )
    exit_status = (
        completed.returncode
        if completed.returncode >= 0
        else 128 + abs(completed.returncode)
    )

    lines = [
        "Measurement source: Python resource.getrusage(RUSAGE_CHILDREN)",
        f"Command being timed: {shlex.join(command)}",
        f"User time (seconds): {user_seconds:.6f}",
        f"System time (seconds): {system_seconds:.6f}",
        f"Percent of CPU this job got: {cpu_percent:.2f}%",
        (
            "Elapsed (wall clock) time (h:mm:ss or m:ss): "
            f"{_format_elapsed(elapsed_seconds)}"
        ),
        f"Maximum resident set size (kbytes): {_max_rss_kbytes(usage):.0f}",
        f"Exit status: {exit_status}",
    ]
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary resource file: {temporary_path}"
        )
    try:
        with temporary_path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return exit_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after '--'")
    return args


def main() -> int:
    args = parse_args()
    return run_and_measure(args.command, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
