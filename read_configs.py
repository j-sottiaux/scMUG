import argparse
import shlex
from pathlib import Path

import yaml


DEFAULT_RED_GLOBAL = "umap"
DEFAULT_RED_LOCAL = "umap"
DEFAULT_N_NEIGHBOUR = 3
GFM_CORRELATION_RULE = "positive"


def bash_export(name, value):
    print(f"export {name}={shlex.quote(str(value))}")


def get_cutoffs(params, n_gfm):
    """Return GFM extension cutoffs as a comma-separated list."""
    if "gfm_extension_cutoffs" in params:
        cutoffs = params["gfm_extension_cutoffs"]
        if isinstance(cutoffs, str):
            values = [float(x) for x in cutoffs.split(",")]
        elif isinstance(cutoffs, list):
            values = [float(x) for x in cutoffs]
        else:
            raise TypeError(
                "'gfm_extension_cutoffs' must be a comma string or a list of floats"
            )
    elif "gfm_extension_cutoff" in params:
        values = [float(params["gfm_extension_cutoff"])] * n_gfm
    elif "cutoffs" in params:
        cutoffs = params["cutoffs"]
        if isinstance(cutoffs, str):
            values = [float(x) for x in cutoffs.split(",")]
        elif isinstance(cutoffs, list):
            values = [float(x) for x in cutoffs]
        else:
            raise TypeError("'cutoffs' must be a comma string or a list of floats")
    elif "cutoff" in params:
        values = [float(params["cutoff"])] * n_gfm
    else:
        raise ValueError(
            "Missing 'gfm_extension_cutoff' or 'gfm_extension_cutoffs'. Use "
            "published dataset-specific GFM extension cutoffs instead of relying "
            "on a hidden default."
        )

    if len(values) != n_gfm:
        raise ValueError(f"Expected {n_gfm} cutoff values, got {len(values)}: {values}")

    return ",".join(str(x) for x in values)


def get_gfm_correlation_rule(params):
    """Require the positive-only GFM extension used by the public scMUG code."""
    rule = params.get("gfm_correlation_rule")
    if rule != GFM_CORRELATION_RULE:
        raise ValueError(
            "'gfm_correlation_rule' must be explicitly set to 'positive' "
            "to match the public scMUG implementation."
        )
    return rule


def ordered_datasets(config):
    """Return dataset names and enforce the experiment's alphabetical contract."""
    if not isinstance(config, dict) or not config:
        raise ValueError("YAML config must contain at least one dataset entry.")

    datasets = list(config.keys())
    expected = sorted(datasets, key=str.casefold)
    if datasets != expected:
        raise ValueError(
            "Datasets must be declared in alphabetical order in the YAML. "
            f"Observed: {datasets}; expected: {expected}"
        )
    return datasets


def resolve_sweep_task(config, task_id, k_min=5, k_max=20):
    """Map a zero-based array task to a one-based dataset index and target k."""
    datasets = ordered_datasets(config)
    if k_min < 1 or k_max < k_min:
        raise ValueError(f"Invalid k range: {k_min}-{k_max}")
    if task_id < 0:
        raise ValueError(f"Sweep task ID must be non-negative, got {task_id}")

    n_k = k_max - k_min + 1
    task_count = len(datasets) * n_k
    if task_id >= task_count:
        raise ValueError(
            f"Invalid sweep task ID {task_id}. Available range: 0-{task_count - 1}"
        )

    dataset_index = task_id // n_k + 1
    cluster_number = k_min + task_id % n_k
    return dataset_index, cluster_number, task_count


def main():
    parser = argparse.ArgumentParser(
        description="Read dataset-specific configuration from YAML for SLURM array jobs."
    )
    parser.add_argument("--config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument(
        "--sweep-task-id",
        type=int,
        help="Zero-based flattened dataset x k task ID.",
    )
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--k-max", type=int, default=20)
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    datasets = ordered_datasets(config)
    sweep_task_count = None
    if args.sweep_task_id is not None:
        config_index, cluster_number, sweep_task_count = resolve_sweep_task(
            config,
            args.sweep_task_id,
            k_min=args.k_min,
            k_max=args.k_max,
        )
    else:
        config_index = args.index
        cluster_number = None

    if config_index < 1 or config_index > len(datasets):
        raise ValueError(
            f"Invalid index {config_index}. Available range: 1-{len(datasets)}"
        )

    dataset = datasets[config_index - 1]
    params = config[dataset]

    required_keys = [
        "published_best_k",
        "ground_truth_k",
        "n_gfm",
        "gfm_correlation_rule",
        "full_alpha",
        "full_beta",
    ]

    missing = [key for key in required_keys if key not in params]
    if missing:
        raise ValueError(f"Missing keys for dataset '{dataset}': {missing}")

    n_gfm = int(params["n_gfm"])
    published_best_k = int(params["published_best_k"])
    ground_truth_k = int(params["ground_truth_k"])
    if published_best_k < 1 or ground_truth_k < 1:
        raise ValueError(
            f"Invalid k metadata for dataset '{dataset}': "
            f"published_best_k={published_best_k}, ground_truth_k={ground_truth_k}"
        )
    if cluster_number is None:
        cluster_number = published_best_k
    cutoffs = get_cutoffs(params, n_gfm)
    correlation_rule = get_gfm_correlation_rule(params)

    bash_export("DATASET", dataset)
    bash_export("CONFIG_INDEX", config_index)
    bash_export("PUBLISHED_BEST_K", published_best_k)
    bash_export("GROUND_TRUTH_K", ground_truth_k)
    bash_export("CLUSTER_NUMBER", cluster_number)
    bash_export("N_GFM", n_gfm)
    bash_export("GFM_CORRELATION_RULE", correlation_rule)
    bash_export("CUTOFFS", cutoffs)
    bash_export("FULL_ALPHA", params["full_alpha"])
    bash_export("FULL_BETA", params["full_beta"])
    bash_export(
        "DMKCN_ALLOW_PSEUDO_COUNTS",
        str(bool(params.get("dmkcn_allow_pseudo_counts", False))).lower(),
    )
    bash_export("N_NEIGHBOUR", DEFAULT_N_NEIGHBOUR)
    bash_export("RED_GLOBAL", DEFAULT_RED_GLOBAL)
    bash_export("RED_LOCAL", DEFAULT_RED_LOCAL)
    if args.sweep_task_id is not None:
        bash_export("K_SWEEP_TASK_ID", args.sweep_task_id)
        bash_export("K_SWEEP_TASK_COUNT", sweep_task_count)
        bash_export("K_MIN", args.k_min)
        bash_export("K_MAX", args.k_max)


if __name__ == "__main__":
    main()
