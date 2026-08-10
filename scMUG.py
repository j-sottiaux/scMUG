import argparse
import hashlib
import json
import math
import os
import tempfile
import numpy as np
import pandas as pd
from model import *
from utils import *
from computation_metrics import (
    ComputationRecorder,
    synchronized_elapsed,
    synchronized_start,
)
from torch import optim
from accelerate import get_mat1, get_mat2
from sklearn.cluster import SpectralClustering


ALPHA_BETA_GRID = [
    (0, 1),
    (0.001, 1),
    (0.01, 1),
    (0.1, 1),
    (1, 1),
    (1, 0.1),
    (1, 0.01),
    (1, 0.001),
    (1, 0),
]


def sha256_lines(values):
    """Stable hash for an ordered collection of identifiers."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_npz_dump(path, **arrays):
    """Write an uncompressed NPZ atomically without pickle payloads."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json_dump(payload, path):
    """Write a JSON artifact atomically within its destination directory."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_csv_dump(frame, path):
    """Write a CSV artifact atomically within its destination directory."""
    target = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_alpha_beta_grid(value):
    """Parse ordered ``alpha:beta`` pairs and reject ambiguous weights."""
    if value is None:
        return None

    pairs = []
    seen = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "alpha/beta grid entries must use 'alpha:beta', for example "
                "'0.01:1,0.1:1,1:1'"
            )
        try:
            alpha, beta = (float(field.strip()) for field in fields)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid alpha/beta pair '{item}': expected numeric values"
            ) from exc
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise argparse.ArgumentTypeError(
                f"invalid alpha/beta pair '{item}': values must be finite"
            )
        if alpha < 0 or beta < 0 or (alpha == 0 and beta == 0):
            raise argparse.ArgumentTypeError(
                f"invalid alpha/beta pair '{item}': weights must be non-negative "
                "and cannot both be zero"
            )
        pair = (alpha, beta)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    if not pairs:
        raise argparse.ArgumentTypeError("alpha/beta grid must contain at least one pair")
    return pairs


def autoencoder_block_b(adata, gene_list, n_sample, epoch, seed):
    """Original scMUG block B: ZINB autoencoder per GFM."""
    set_seed(seed)
    batchSize = int(max(min(2 ** (n_sample**0.5 // 8 + 1), 64), 4))
    mask = adata.var_names.isin(gene_list)
    input_data, target_data = adata.X[:, mask], adata.X[:, mask]
    input_dim = input_data.shape[1]
    data_loader = get_data_loader(input_data, target_data, batch_size=batchSize)
    model = Autoencoder([input_dim, 512, 128, 32]).to(device)
    criterion = ZINBLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, data_loader, epoch, criterion, optimizer)
    return get_encoded_output(model, data_loader)


def dmkcn_adapter_block_b(
    adata,
    gene_list,
    cluster_number,
    seed,
    zinb_on_counts=True,
    full_training=True,
    allow_pseudo_counts=False,
    projections=None,
    embedding_dims=None,
    primary_projection="spectral_dense",
    primary_d=32,
    graph_neighbors=30,
    lambda1=None,
    lambda2=None,
    lambda3=None,
    include_kernel_representation=False,
):
    """DMKCN replacement for scMUG block B. Imported lazily to keep AE path usable."""
    from dmkcn.integration import dmkcn_block_b

    return dmkcn_block_b(
        adata,
        gene_list,
        n_clusters=cluster_number,
        d=primary_d,
        seed=seed,
        full_training=full_training,
        zinb_on_counts=zinb_on_counts,
        allow_pseudo_counts=allow_pseudo_counts,
        projections=projections,
        embedding_dims=embedding_dims,
        primary_projection=primary_projection,
        graph_neighbors=graph_neighbors,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        return_artifacts=True,
        include_kernel_representation=include_kernel_representation,
    )


def clustering_metrics(y_true, labels):
    return {
        "nmi": float(calc_nmi(y_true, labels)),
        "ari": float(calc_ari(y_true, labels)),
        "acc": float(calc_acc(y_true, labels)),
    }


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def run():
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", default="muraro", type=str)
    parser.add_argument("--cluster_number", default=None, type=int)
    parser.add_argument(
        "--seeds",
        default="1111,2222,3333,4444,5555,6666,7777,8888,9999,10000",
        type=str,
    )
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--n_gfm", default=5, type=int)
    parser.add_argument("--cutoffs", default="0.14,0.14,0.15,0.14,0.14", type=str)
    parser.add_argument(
        "--gfm-correlation-rule",
        default="positive",
        choices=["positive"],
        help="Positive-only GFM extension rule used by the public scMUG code.",
    )
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--n_neighbour", default=3, type=int)
    parser.add_argument("--kmeans_times", default=20, type=int)
    parser.add_argument("--red_global", type=str)
    parser.add_argument("--red_local", type=str)
    parser.add_argument(
        "--thread-num", "--thread_num", dest="thread_num", default=16, type=int
    )
    parser.add_argument(
        "--block-b",
        default="dmkcn",
        choices=["autoencoder", "dmkcn"],
        help="Block B representation generator. 'autoencoder' is original scMUG; 'dmkcn' is the adapter branch.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        type=str,
        help="Suffix used for output files. Defaults to the selected --block-b.",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        type=str,
        help="Directory where output files are written.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Unique phase/k/run prefix for artifact filenames.",
    )
    parser.add_argument("--experiment-id", default="main")
    parser.add_argument("--run-id", default="local")
    parser.add_argument("--canonical-alpha", default=1.0, type=float)
    parser.add_argument("--canonical-beta", default=1.0, type=float)
    parser.add_argument(
        "--alpha-beta-grid",
        type=parse_alpha_beta_grid,
        default=None,
        help=(
            "Ordered alpha:beta pairs evaluated in block D. By default, use the "
            "nine-pair grid from the public scMUG implementation."
        ),
    )
    parser.add_argument(
        "--computation-outfile",
        default=None,
        help="Run-specific CSV receiving wall-time and process-memory metrics.",
    )
    parser.add_argument(
        "--dmkcn-zinb-on-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only for --block-b dmkcn. True uses raw counts as ZINB target; false mirrors scMUG scaled-target behavior.",
    )
    parser.add_argument(
        "--dmkcn-full-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only for --block-b dmkcn. Disable to run DMKCN pretraining only.",
    )
    parser.add_argument(
        "--dmkcn-allow-pseudo-counts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly round non-integer raw values for the DMKCN ZINB target.",
    )
    parser.add_argument(
        "--dmkcn-projections",
        default="spectral_dense",
        help="Comma-separated K projections to save: spectral_dense, spectral_knn, svd_raw, svd_l2, svd_zscore.",
    )
    parser.add_argument(
        "--dmkcn-embedding-dims",
        default="32",
        help="Comma-separated embedding dimensions generated from the same trained K.",
    )
    parser.add_argument(
        "--dmkcn-primary-projection",
        default="spectral_dense",
        help="Projection used by the full scMUG C/D pipeline in this invocation.",
    )
    parser.add_argument(
        "--dmkcn-primary-d",
        default=32,
        type=int,
        help="Embedding dimension used by the full scMUG C/D pipeline.",
    )
    parser.add_argument(
        "--dmkcn-graph-neighbors",
        default=30,
        type=int,
        help="Weighted kNN graph size for the spectral_knn projection.",
    )
    parser.add_argument(
        "--dmkcn-evaluate-exits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate K and each saved embedding against labels for diagnostics only.",
    )
    parser.add_argument(
        "--dmkcn-lambda-config",
        default=None,
        help=(
            "YAML file containing dataset-specific lambda1/lambda2/lambda3 values. "
            "When omitted, the historical trainer defaults are preserved."
        ),
    )
    parser.add_argument(
        "--dmkcn-lambda-config-id",
        default=None,
        help=(
            "Optional candidate ID from --dmkcn-lambda-config. This reuses a "
            "validated candidate without changing the source campaign's dataset map."
        ),
    )
    parser.add_argument(
        "--dmkcn-export-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export one float32 (n_gfm, n_cells, n_cells) NPZ per seed, plus "
            "cell and GFM membership metadata. Disabled by default."
        ),
    )

    args = parser.parse_args()

    dbname = args.dataset
    seeds = [int(_) for _ in args.seeds.split(",")]
    repeat = args.repeat
    n_gfm = args.n_gfm
    cluster_number = args.cluster_number
    cutoffs = [float(_) for _ in args.cutoffs.split(",")]
    gfm_correlation_rule = args.gfm_correlation_rule
    epoch = args.epoch
    n_neighbour = args.n_neighbour
    kmeans_times = args.kmeans_times
    red_global = args.red_global
    red_local = args.red_local
    block_b = args.block_b
    lambda_triplet = None
    if args.dmkcn_lambda_config_id is not None and args.dmkcn_lambda_config is None:
        raise ValueError(
            "--dmkcn-lambda-config-id requires --dmkcn-lambda-config."
        )
    if args.dmkcn_export_kernels and block_b != "dmkcn":
        raise ValueError("--dmkcn-export-kernels is only valid with --block-b dmkcn.")
    if args.dmkcn_export_kernels and args.output_stem is None:
        raise ValueError("--dmkcn-export-kernels requires a unique --output-stem.")
    if args.dmkcn_lambda_config is not None:
        if block_b != "dmkcn":
            raise ValueError(
                "--dmkcn-lambda-config is only valid with --block-b dmkcn."
            )
        from dmkcn.lambda_config import load_lambda_triplet

        lambda_triplet = load_lambda_triplet(
            args.dmkcn_lambda_config,
            dbname,
            candidate_id=args.dmkcn_lambda_config_id,
        )
    if block_b == "autoencoder":
        lambda_metadata = {
            "lambda_config_id": None,
            "lambda1": None,
            "lambda2": None,
            "lambda3": None,
            "campaign_id": None,
            "source_file": None,
        }
    elif lambda_triplet is None:
        lambda_metadata = {
            "lambda_config_id": "legacy_python_defaults",
            "lambda1": None,
            "lambda2": None,
            "lambda3": None,
            "campaign_id": None,
            "source_file": None,
        }
    else:
        lambda_metadata = lambda_triplet.as_dict()
    output_tag = args.output_tag or block_b
    output_dir = args.output_dir
    output_stem = args.output_stem
    if output_stem is not None and (
        output_stem != os.path.basename(output_stem) or output_stem in {".", ".."}
    ):
        raise ValueError("--output-stem must be a filename prefix, not a path")
    thread_num = args.thread_num
    dmkcn_projections = [
        value.strip() for value in args.dmkcn_projections.split(",") if value.strip()
    ]
    dmkcn_embedding_dims = [
        int(value) for value in args.dmkcn_embedding_dims.split(",") if value.strip()
    ]
    if args.dmkcn_primary_projection not in dmkcn_projections:
        dmkcn_projections.append(args.dmkcn_primary_projection)
    if args.dmkcn_primary_d not in dmkcn_embedding_dims:
        dmkcn_embedding_dims.append(args.dmkcn_primary_d)
    alpha_beta_grid = args.alpha_beta_grid or ALPHA_BETA_GRID
    if not any(
        np.isclose(alpha, args.canonical_alpha)
        and np.isclose(beta, args.canonical_beta)
        for alpha, beta in alpha_beta_grid
    ):
        raise ValueError(
            "The canonical alpha/beta pair must be present in --alpha-beta-grid: "
            f"{args.canonical_alpha}, {args.canonical_beta}"
        )

    os.makedirs(output_dir, exist_ok=True)

    pipeline_started = synchronized_start()
    data_loading_started = synchronized_start()
    expr_df, cell_type = load_data(dbname)
    data_loading_seconds = synchronized_elapsed(data_loading_started)
    print(f"\nDatabase: {dbname}\tCells: {expr_df.shape[0]}\tGenes: {expr_df.shape[1]}")
    print(f"Block B: {block_b}\tOutput tag: {output_tag}")
    print(f"Alpha/beta grid: {alpha_beta_grid}")
    if block_b == "dmkcn":
        print(f"DMKCN lambda configuration: {lambda_metadata}")
    source_cell_ids = [str(value) for value in expr_df.index]
    if len(set(source_cell_ids)) != len(source_cell_ids):
        raise ValueError("Source expression matrix contains duplicated cell identifiers.")
    if len(source_cell_ids) != len(cell_type):
        raise ValueError(
            "Source cell identifier and label lengths differ: "
            f"{len(source_cell_ids)} != {len(cell_type)}."
        )
    cell_ids_sha256 = sha256_lines(source_cell_ids)
    source_gene_count = int(expr_df.shape[1])
    preprocessing_started = synchronized_start()
    expr_df = expr_df.astype(float)
    adata = preprocess(expr_df=expr_df, cell_type=cell_type, highly_genes=8000)
    y = lab2fac(adata.obs["cell_type"].to_numpy())
    preprocessing_seconds = synchronized_elapsed(preprocessing_started)
    n_sample = adata.X.shape[0]
    if cluster_number is None:
        cluster_number = len(set(y))

    pipeline_name = "scMUG" if block_b == "autoencoder" else "scMUG-DMKCN"
    recorder = ComputationRecorder(
        args.computation_outfile,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        dataset=dbname,
        k=cluster_number,
        pipeline=pipeline_name,
        n_gfm=n_gfm,
        lambda_config_id=lambda_metadata["lambda_config_id"],
        lambda1=lambda_metadata["lambda1"],
        lambda2=lambda_metadata["lambda2"],
        lambda3=lambda_metadata["lambda3"],
    )
    recorder.set_dimensions(
        n_cells=n_sample,
        n_genes=source_gene_count,
        n_hvg=int(adata.n_vars),
    )
    recorder.record("data_loading", data_loading_seconds)
    recorder.record("preprocessing", preprocessing_seconds)
    scientific_seed_stages = {
        "gfm_extension",
        "ae_training" if block_b == "autoencoder" else "dmkcn_block_b_total",
        "block_c_local",
        "block_c_global_reduction",
        "block_c_global",
        "block_d_spectral",
    }
    scientific_seed_stages_without_d = scientific_seed_stages - {
        "block_d_spectral"
    }

    def canonical_elapsed(seed=None):
        base = recorder.elapsed_sum(scientific_seed_stages_without_d, seed=seed)
        canonical_d = sum(
            row["elapsed_seconds"]
            for row in recorder.rows
            if row["stage"] == "block_d_spectral"
            and (seed is None or row["seed"] == seed)
            and np.isclose(row["alpha"], args.canonical_alpha)
            and np.isclose(row["beta"], args.canonical_beta)
        )
        return float(base + canonical_d)

    predictions = []
    latents = []
    latent_variants = {}
    kernel_paths = {}
    gfm_membership_reference = None

    if args.dmkcn_export_kernels:
        export_prefix = f"{output_stem}_{output_tag}"
        kernels_dir = os.path.join(output_dir, f"{export_prefix}_kernels")
        cell_metadata_path = os.path.join(
            output_dir, f"{export_prefix}_cell_metadata.csv"
        )
        gfm_membership_path = os.path.join(
            output_dir, f"{export_prefix}_gfm_membership.json"
        )
        for target in (kernels_dir, cell_metadata_path, gfm_membership_path):
            if os.path.exists(target):
                raise FileExistsError(
                    f"Refusing to overwrite existing kernel export target: {target}"
                )
    else:
        kernels_dir = None
        cell_metadata_path = None
        gfm_membership_path = None

    diagnostics_name = (
        f"{output_stem}_{output_tag}_diagnostics.jsonl"
        if output_stem
        else f"dmkcn_diagnostics_{output_tag}.jsonl"
    )
    diagnostics_path = os.path.join(output_dir, diagnostics_name)
    if block_b == "dmkcn":
        with open(diagnostics_path, "w", encoding="utf-8"):
            pass

    result_name = (
        f"{output_stem}_{output_tag}_raw.txt"
        if output_stem
        else f"{dbname}s_{output_tag}.txt"
    )
    result_path = os.path.join(output_dir, result_name)
    f = open(result_path, "w", encoding="utf-8")

    for seed in seeds:
        print(f"\nSeed: {seed}\n")
        seed_started = recorder.start()
        latent_val = None
        seed_variant_latents = {}
        seed_kernel_representations = []
        seed_gfm_membership = []
        for i, t in enumerate(cutoffs):
            print(f"GFM: {i + 1}")
            gfm_started = recorder.start()
            with open(f"./GFMs/{dbname}/{i + 1}.txt", "r") as fg:
                gfm = set(
                    [
                        s.replace("\n", "").replace("\t", "")
                        for s in fg.readlines()
                        if s[0] != "#"
                    ]
                )
            gfm = list(gfm & set(adata.var.index))
            gene_list = extend_gfm(
                adata,
                gfm,
                t,
                d=3,
                correlation_rule=gfm_correlation_rule,
            )
            seed_gfm_membership.append(
                {
                    "gfm_index": int(i + 1),
                    "seed_genes": sorted(str(value) for value in gfm),
                    "extended_genes": sorted(str(value) for value in gene_list),
                    "cutoff": float(t),
                    "correlation_rule": gfm_correlation_rule,
                }
            )
            seed_gfm_membership[-1]["seed_genes_sha256"] = sha256_lines(
                seed_gfm_membership[-1]["seed_genes"]
            )
            seed_gfm_membership[-1]["extended_genes_sha256"] = sha256_lines(
                seed_gfm_membership[-1]["extended_genes"]
            )
            recorder.finish(
                gfm_started,
                "gfm_extension",
                seed=seed,
                gfm_index=i + 1,
                gfm_gene_count=len(gene_list),
            )

            if block_b == "autoencoder":
                block_b_started = recorder.start()
                latent = autoencoder_block_b(
                    adata=adata,
                    gene_list=gene_list,
                    n_sample=n_sample,
                    epoch=epoch,
                    seed=seed,
                )
                recorder.finish(
                    block_b_started,
                    "ae_training",
                    seed=seed,
                    gfm_index=i + 1,
                    gfm_gene_count=len(gene_list),
                )
            elif block_b == "dmkcn":
                set_seed(seed)
                block_b_started = recorder.start()
                artifacts = dmkcn_adapter_block_b(
                    adata=adata,
                    gene_list=gene_list,
                    cluster_number=cluster_number,
                    seed=seed,
                    zinb_on_counts=args.dmkcn_zinb_on_counts,
                    full_training=args.dmkcn_full_training,
                    allow_pseudo_counts=args.dmkcn_allow_pseudo_counts,
                    projections=dmkcn_projections,
                    embedding_dims=dmkcn_embedding_dims,
                    primary_projection=args.dmkcn_primary_projection,
                    primary_d=args.dmkcn_primary_d,
                    graph_neighbors=args.dmkcn_graph_neighbors,
                    lambda1=(
                        None if lambda_triplet is None else lambda_triplet.lambda1
                    ),
                    lambda2=(
                        None if lambda_triplet is None else lambda_triplet.lambda2
                    ),
                    lambda3=(
                        None if lambda_triplet is None else lambda_triplet.lambda3
                    ),
                    include_kernel_representation=args.dmkcn_export_kernels,
                )
                recorder.finish(
                    block_b_started,
                    "dmkcn_block_b_total",
                    seed=seed,
                    gfm_index=i + 1,
                    gfm_gene_count=len(gene_list),
                )
                for stage, elapsed_seconds in artifacts["timings"].items():
                    recorder.record(
                        stage,
                        elapsed_seconds,
                        seed=seed,
                        gfm_index=i + 1,
                        gfm_gene_count=len(gene_list),
                    )
                diagnostic_record = {
                    "experiment_id": args.experiment_id,
                    "run_id": args.run_id,
                    "dataset": dbname,
                    "seed": int(seed),
                    "gfm_index": int(i + 1),
                    "gfm_seed_gene_count": int(len(gfm)),
                    "gfm_extended_gene_count": int(len(gene_list)),
                    "gfm_correlation_rule": gfm_correlation_rule,
                    "cluster_number": int(cluster_number),
                    "lambda_configuration": lambda_metadata,
                    "primary_key": artifacts["primary_key"],
                    "timings": artifacts["timings"],
                    **artifacts["diagnostics"],
                }
                if args.dmkcn_evaluate_exits:
                    exit_metrics = {
                        "K_raw": clustering_metrics(y, artifacts["labels_K_raw"])
                    }
                    for variant_key, variant_embedding in artifacts[
                        "embeddings"
                    ].items():
                        variant_labels, _ = c_kmeans(
                            variant_embedding,
                            cluster_number,
                            n_init=20,
                            random_state=seed + i,
                        )
                        exit_metrics[variant_key] = clustering_metrics(
                            y, variant_labels
                        )
                    diagnostic_record["exit_metrics"] = exit_metrics
                append_jsonl(diagnostics_path, diagnostic_record)

                if args.dmkcn_export_kernels:
                    seed_kernel_representations.append(
                        artifacts.pop("kernel_representation")
                    )

                for variant_key, variant_embedding in artifacts[
                    "embeddings"
                ].items():
                    shaped = variant_embedding.reshape(
                        (variant_embedding.shape[0], 1, -1)
                    )
                    if variant_key not in seed_variant_latents:
                        seed_variant_latents[variant_key] = shaped
                    else:
                        seed_variant_latents[variant_key] = np.concatenate(
                            (seed_variant_latents[variant_key], shaped), axis=1
                        )
                latent = artifacts["embeddings"][artifacts["primary_key"]]
            else:
                raise ValueError(f"Unknown --block-b: {block_b}")

            latent = latent.reshape((latent.shape[0], 1, -1))
            if i == 0:
                latent_val = latent
            else:
                latent_val = np.concatenate((latent_val, latent), axis=1)

        if args.dmkcn_export_kernels:
            if len(seed_kernel_representations) != n_gfm:
                raise ValueError(
                    f"Seed {seed}: expected {n_gfm} kernel matrices, got "
                    f"{len(seed_kernel_representations)}."
                )
            if gfm_membership_reference is None:
                gfm_membership_reference = seed_gfm_membership
            elif seed_gfm_membership != gfm_membership_reference:
                raise ValueError(
                    "Extended GFM membership changed between seeds; refusing to "
                    "write an ambiguous campaign artifact."
                )
            kernel_stack = np.stack(seed_kernel_representations).astype(
                np.float32, copy=False
            )
            expected_shape = (n_gfm, n_sample, n_sample)
            if kernel_stack.shape != expected_shape:
                raise ValueError(
                    f"Seed {seed}: expected kernel shape {expected_shape}, got "
                    f"{kernel_stack.shape}."
                )
            kernel_path = os.path.join(
                kernels_dir, f"seed{int(seed)}_kernel_representations.npz"
            )
            kernel_export_started = recorder.start()
            atomic_npz_dump(
                kernel_path,
                kernels=kernel_stack,
                gfm_indices=np.arange(1, n_gfm + 1, dtype=np.int16),
                seed=np.asarray(int(seed), dtype=np.int64),
                cell_ids_sha256=np.asarray(cell_ids_sha256),
            )
            recorder.finish(
                kernel_export_started,
                "kernel_export_io",
                seed=seed,
            )
            kernel_paths[str(int(seed))] = kernel_path
            del kernel_stack
            seed_kernel_representations.clear()

        latents.append(latent_val)
        if block_b == "dmkcn":
            for variant_key, variant_latent in seed_variant_latents.items():
                latent_variants.setdefault(variant_key, []).append(variant_latent)

        # local feature
        block_c_local_started = recorder.start()
        dist = np.zeros(shape=(n_gfm, n_sample, n_sample))
        for c in range(n_gfm):
            X = latent_val[:, c, :]

            if not np.isfinite(X).all():
                print(f"[WARNING] NaN/Inf detected before local reducer, GFM={c + 1}")
                print("NaN count:", np.isnan(X).sum())
                print("Inf count:", np.isinf(X).sum())
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            latent_val_c = reducer(red_local)(X)

            for i in range(n_sample):
                for j in range(i + 1, n_sample):
                    dist[c, i, j] = dist[c, j, i] = np.linalg.norm(
                        latent_val_c[i] - latent_val_c[j]
                    )

        neighbourDist = np.array(
            [
                np.array(
                    [
                        np.sum(
                            dis[
                                i,
                                np.argpartition(dis[i], n_neighbour + 1)[
                                    1 : n_neighbour + 1
                                ],
                            ]
                        )
                        / n_neighbour
                        for i in range(n_sample)
                    ]
                )
                for dis in dist
            ]
        )

        neighbourDistScore = (
            np.array(
                [
                    np.array(
                        [
                            1
                            / np.log(
                                np.var(
                                    dis[
                                        i,
                                        np.argpartition(dis[i], n_neighbour + 1)[
                                            1 : n_neighbour + 1
                                        ],
                                    ]
                                )
                                + np.exp(1)
                            )
                            for i in range(n_sample)
                        ]
                    )
                    for dis in dist
                ]
            )
            ** 0.5
        )

        mat2 = get_mat2(n_sample, neighbourDist, dist, neighbourDistScore)
        recorder.finish(
            block_c_local_started,
            "block_c_local",
            seed=seed,
        )

        global_reduction_started = recorder.start()
        global_views = []
        for c in range(n_gfm):
            z = latent_val[:, c, :].reshape(latent_val.shape[0], -1)
            if not np.isfinite(z).all():
                print(
                    f"[WARNING] NaN/Inf detected before global reducer, GFM={c + 1}"
                )
                print("NaN count:", np.isnan(z).sum())
                print("Inf count:", np.isinf(z).sum())
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            global_views.append(reducer(red_global)(z))
        recorder.finish(
            global_reduction_started,
            "block_c_global_reduction",
            seed=seed,
        )

        for r in range(repeat):
            print(f"\nRound {r}")
            repeat_seed = seed + r
            set_seed(repeat_seed)
            # global feature
            block_c_global_started = recorder.start()
            pred = np.zeros(shape=(kmeans_times, n_gfm, latent_val.shape[0])).astype(
                int
            )
            score = np.zeros(shape=(kmeans_times, n_gfm, latent_val.shape[0]))
            for c in range(n_gfm):
                z = global_views[c]

                for t in range(kmeans_times):
                    kmeans_seed = block_c_kmeans_seed(
                        seed,
                        r,
                        c,
                        t,
                        n_gfm,
                        kmeans_times,
                    )
                    pred_z, score_z = c_kmeans(
                        z, cluster_number, n_init=10, random_state=kmeans_seed
                    )
                    pred[t, c, :] = pred_z
                    score_z = np.sort(score_z, axis=1)
                    denom = score_z[:, 1] + score_z[:, 0]
                    denom = np.where(denom == 0, 1e-12, denom)
                    score[t, c, :] = (
                        ((score_z[:, 1] - score_z[:, 0]) / denom) ** 0.5
                        / kmeans_times
                        / n_gfm
                    )

            mat1 = get_mat1(
                pred, n_sample, kmeans_times, n_gfm, cluster_number, score, thread_num
            )
            mean_mat1 = np.mean(mat1)
            if mean_mat1 > 1e-12:
                mat1 = mat1 / mean_mat1
            else:
                print("[WARNING] mat1 mean is zero; skipping mat1 normalization")
            recorder.finish(
                block_c_global_started,
                "block_c_global",
                seed=seed,
                repeat=r,
            )

            for alpha, beta in alpha_beta_grid:
                mat = mat1 * alpha + mat2 * beta
                block_d_started = recorder.start()
                Spec = SpectralClustering(
                    n_clusters=cluster_number,
                    random_state=repeat_seed,
                    affinity="precomputed",
                )
                labels = Spec.fit_predict(mat)
                recorder.finish(
                    block_d_started,
                    "block_d_spectral",
                    seed=seed,
                    repeat=r,
                    alpha=alpha,
                    beta=beta,
                )
                print(
                    f"dbname:{dbname}\tround:{seed}\talpha:{round(alpha, 3)}\tbeta:{round(beta, 3)}\t",
                    end="",
                )
                _ = benchmark(y, labels)
                p12 = labels
                predictions.append(p12)
                f.write(
                    f"dbname:{dbname}\tround:{seed}\talpha:{alpha}\tbeta:{beta}\t{benchmark(y, p12, False)}\n"
                )

        seed_wall_seconds = synchronized_elapsed(seed_started)
        recorder.record("seed_wall_total", seed_wall_seconds, seed=seed)
        recorder.record(
            "seed_total",
            recorder.elapsed_sum(scientific_seed_stages, seed=seed),
            seed=seed,
        )
        recorder.record(
            "canonical_seed_total",
            canonical_elapsed(seed=seed),
            seed=seed,
            alpha=args.canonical_alpha,
            beta=args.canonical_beta,
        )

    pipeline_wall_seconds = synchronized_elapsed(pipeline_started)
    recorder.record("pipeline_wall_total", pipeline_wall_seconds)
    recorder.record(
        "pipeline_total",
        data_loading_seconds
        + preprocessing_seconds
        + recorder.elapsed_sum(scientific_seed_stages),
    )
    recorder.record(
        "canonical_pipeline_total",
        data_loading_seconds + preprocessing_seconds + canonical_elapsed(),
        alpha=args.canonical_alpha,
        beta=args.canonical_beta,
    )
    recorder.flush()
    f.close()

    predictions_name = (
        f"{output_stem}_{output_tag}_predictions.joblib"
        if output_stem
        else f"pred_label_{output_tag}.joblib"
    )
    latents_name = (
        f"{output_stem}_{output_tag}_latents.joblib"
        if output_stem
        else f"latents_{output_tag}.joblib"
    )
    predictions_path = os.path.join(output_dir, predictions_name)
    latents_path = os.path.join(output_dir, latents_name)
    atomic_joblib_dump(predictions, predictions_path)
    atomic_joblib_dump(latents, latents_path)

    if block_b == "dmkcn":
        if args.dmkcn_export_kernels:
            if sorted(kernel_paths) != sorted(str(int(seed)) for seed in seeds):
                raise ValueError(
                    "Kernel export is incomplete: expected one artifact for every seed."
                )
            if gfm_membership_reference is None:
                raise ValueError("GFM membership metadata was not collected.")
            atomic_csv_dump(
                pd.DataFrame(
                    {
                        "cell_position": np.arange(n_sample, dtype=int),
                        "cell_id": source_cell_ids,
                        "cell_type": [str(value) for value in cell_type],
                        "encoded_label": np.asarray(y, dtype=int),
                    }
                ),
                cell_metadata_path,
            )
            atomic_json_dump(
                {
                    "schema_version": 1,
                    "dataset": dbname,
                    "n_cells": int(n_sample),
                    "n_gfm": int(n_gfm),
                    "cell_ids_sha256": cell_ids_sha256,
                    "gfm_membership": gfm_membership_reference,
                },
                gfm_membership_path,
            )
        variant_paths = {}
        for variant_key, values in sorted(latent_variants.items()):
            variant_path = os.path.join(
                output_dir,
                (
                    f"{output_stem}_{output_tag}_latents__{variant_key}.joblib"
                    if output_stem
                    else f"latents_{output_tag}__{variant_key}.joblib"
                ),
            )
            atomic_joblib_dump(values, variant_path)
            variant_paths[variant_key] = variant_path
        manifest = {
            "experiment_id": args.experiment_id,
            "run_id": args.run_id,
            "dataset": dbname,
            "cluster_number": int(cluster_number),
            "alpha_beta_grid": [list(pair) for pair in alpha_beta_grid],
            "canonical_alpha": float(args.canonical_alpha),
            "canonical_beta": float(args.canonical_beta),
            "lambda_configuration": lambda_metadata,
            "output_tag": output_tag,
            "primary_key": f"{args.dmkcn_primary_projection}_d{args.dmkcn_primary_d}",
            "primary_path": latents_path,
            "legacy_primary_path": latents_path,
            "variant_paths": variant_paths,
            "diagnostics_path": diagnostics_path,
            "kernel_export": {
                "enabled": bool(args.dmkcn_export_kernels),
                "paths_by_seed": kernel_paths,
                "cell_metadata_path": cell_metadata_path,
                "gfm_membership_path": gfm_membership_path,
                "cell_ids_sha256": (
                    cell_ids_sha256 if args.dmkcn_export_kernels else None
                ),
                "format": (
                    "npz_float32_gfm_cells_cells"
                    if args.dmkcn_export_kernels
                    else None
                ),
            },
        }
        manifest_name = (
            f"{output_stem}_{output_tag}_manifest.json"
            if output_stem
            else f"latents_{output_tag}_manifest.json"
        )
        atomic_json_dump(manifest, os.path.join(output_dir, manifest_name))


if __name__ == "__main__":
    run()
