import argparse
import shlex
from pathlib import Path

import yaml


DEFAULT_CUTOFF = 0.14
DEFAULT_RED_GLOBAL = "umap"
DEFAULT_RED_LOCAL = "umap"
DEFAULT_N_NEIGHBOUR = 3


def bash_export(name, value):
    print(f"export {name}={shlex.quote(str(value))}")


def main():
    parser = argparse.ArgumentParser(
        description="Read dataset-specific configuration from YAML for SLURM array jobs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or not config:
        raise ValueError("YAML config must contain at least one dataset entry.")

    datasets = list(config.keys())

    if args.index < 1 or args.index > len(datasets):
        raise ValueError(
            f"Invalid index {args.index}. Available range: 1-{len(datasets)}"
        )

    dataset = datasets[args.index - 1]
    params = config[dataset]

    required_keys = [
        "cluster_number",
        "n_gfm",
        "full_alpha",
        "full_beta",
    ]

    missing = [key for key in required_keys if key not in params]
    if missing:
        raise ValueError(f"Missing keys for dataset '{dataset}': {missing}")

    n_gfm = int(params["n_gfm"])
    cutoffs = ",".join([str(DEFAULT_CUTOFF)] * n_gfm)

    bash_export("DATASET", dataset)
    bash_export("CLUSTER_NUMBER", params["cluster_number"])
    bash_export("N_GFM", n_gfm)
    bash_export("CUTOFFS", cutoffs)
    bash_export("FULL_ALPHA", params["full_alpha"])
    bash_export("FULL_BETA", params["full_beta"])
    bash_export("N_NEIGHBOUR", DEFAULT_N_NEIGHBOUR)
    bash_export("RED_GLOBAL", DEFAULT_RED_GLOBAL)
    bash_export("RED_LOCAL", DEFAULT_RED_LOCAL)


if __name__ == "__main__":
    main()
