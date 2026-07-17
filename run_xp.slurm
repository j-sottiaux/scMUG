#!/bin/bash
#SBATCH --job-name=xp_all
#SBATCH --output=logs/result_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=L40S:1
#SBATCH --partition=gpuai
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr
#SBATCH --array=2,3%2

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
GCORR_DIR="${GCORR_DIR:-${PROJECT_DIR}/gcorr}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/configs/data.yaml}"
K_SWEEP="${K_SWEEP:-false}"
K_MIN="${K_MIN:-5}"
K_MAX="${K_MAX:-20}"

echo "=== Job started on $(hostname) at $(date) ==="
echo "=== SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} ==="

module purge
module load pytorch/2.0.1/gpu

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"

echo "=== Load dataset configuration ==="
READ_CONFIG_ARGS=(--config "${CONFIG_FILE}")
if [ "${K_SWEEP}" = "true" ]; then
    READ_CONFIG_ARGS+=(
        --sweep-task-id "${SLURM_ARRAY_TASK_ID}"
        --k-min "${K_MIN}"
        --k-max "${K_MAX}"
    )
else
    CONFIG_INDEX="${CONFIG_INDEX:-${SLURM_ARRAY_TASK_ID}}"
    READ_CONFIG_ARGS+=(--index "${CONFIG_INDEX}")
fi
load_shell_exports \
    python3 "${PROJECT_DIR}/read_configs.py" "${READ_CONFIG_ARGS[@]}"

echo "Dataset: ${DATASET}"
echo "Cluster number: ${CLUSTER_NUMBER}"
echo "Published best k: ${PUBLISHED_BEST_K}"
echo "Ground-truth k: ${GROUND_TRUTH_K}"
echo "N_GFM: ${N_GFM}"
echo "GFM correlation rule: ${GFM_CORRELATION_RULE}"
echo "Cutoffs: ${CUTOFFS}"
echo "Full alpha: ${FULL_ALPHA}"
echo "Full beta: ${FULL_BETA}"
echo "N_NEIGHBOUR: ${N_NEIGHBOUR}"
echo "RED_GLOBAL: ${RED_GLOBAL}"
echo "RED_LOCAL: ${RED_LOCAL}"
echo "DMKCN_ALLOW_PSEUDO_COUNTS: ${DMKCN_ALLOW_PSEUDO_COUNTS}"

DMKCN_ADAPTER_PILOT="${DMKCN_ADAPTER_PILOT:-false}"
SEEDS="${SEEDS:-1111,2222,3333,4444,5555,6666,7777,8888,9999,10000}"
REPEAT="${REPEAT:-3}"
KMEANS_TIMES="${KMEANS_TIMES:-20}"
EPOCH="${EPOCH:-50}"
DMKCN_PROJECTIONS="${DMKCN_PROJECTIONS:-spectral_dense}"
DMKCN_EMBEDDING_DIMS="${DMKCN_EMBEDDING_DIMS:-32}"
DMKCN_PRIMARY_PROJECTION="${DMKCN_PRIMARY_PROJECTION:-spectral_dense}"
DMKCN_PRIMARY_D="${DMKCN_PRIMARY_D:-32}"
DMKCN_GRAPH_NEIGHBORS="${DMKCN_GRAPH_NEIGHBORS:-30}"
DMKCN_EVALUATE_EXITS="${DMKCN_EVALUATE_EXITS:-false}"
DMKCN_EVALUATE_VARIANTS="${DMKCN_EVALUATE_VARIANTS:-false}"
CD_VALIDATION_ALPHA_BETA_GRID="${CD_VALIDATION_ALPHA_BETA_GRID:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
EXPERIMENT_PHASE="${EXPERIMENT_PHASE:-main}"
ALPHA_BETA_MODE="${ALPHA_BETA_MODE:-full}"
SCMUG_ALPHA_BETA_GRID="${SCMUG_ALPHA_BETA_GRID:-}"
RUN_ABLATIONS="${RUN_ABLATIONS:-true}"
RUN_ALL_ALPHA_BETA_SUMMARY="${RUN_ALL_ALPHA_BETA_SUMMARY:-true}"
RUN_STATISTICS="${RUN_STATISTICS:-true}"

if [ "${DMKCN_ADAPTER_PILOT}" = "true" ]; then
    SEEDS="1111,2222,3333"
    REPEAT=1
    DMKCN_PROJECTIONS="spectral_dense,spectral_knn,svd_raw,svd_l2,svd_zscore"
    DMKCN_EMBEDDING_DIMS="9,16,18,32"
    DMKCN_EVALUATE_EXITS=true
    DMKCN_EVALUATE_VARIANTS=true
    EXPERIMENT_ID="${EXPERIMENT_ID:-adapter_pilot}"
else
    EXPERIMENT_ID="${EXPERIMENT_ID:-main}"
fi
case "${ALPHA_BETA_MODE}" in
    full)
        SCMUG_ALPHA_BETA_GRID="0:1,0.001:1,0.01:1,0.1:1,1:1,1:0.1,1:0.01,1:0.001,1:0"
        ;;
    canonical)
        SCMUG_ALPHA_BETA_GRID="${FULL_ALPHA}:${FULL_BETA}"
        ;;
    custom)
        if [ -z "${SCMUG_ALPHA_BETA_GRID}" ]; then
            echo "SCMUG_ALPHA_BETA_GRID is required when ALPHA_BETA_MODE=custom." >&2
            exit 2
        fi
        ;;
    *)
        echo "ALPHA_BETA_MODE must be full, canonical or custom." >&2
        exit 2
        ;;
esac
for boolean_name in RUN_ABLATIONS RUN_ALL_ALPHA_BETA_SUMMARY RUN_STATISTICS; do
    boolean_value="${!boolean_name}"
    if [ "${boolean_value}" != "true" ] && [ "${boolean_value}" != "false" ]; then
        echo "${boolean_name} must be true or false, got ${boolean_value}." >&2
        exit 2
    fi
done
if [ "${RUN_STATISTICS}" = "true" ] && [ "${RUN_ABLATIONS}" != "true" ]; then
    echo "RUN_STATISTICS=true requires RUN_ABLATIONS=true." >&2
    exit 2
fi
if [[ ! "${EXPERIMENT_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "EXPERIMENT_ID must contain only letters, digits, dot, underscore or hyphen." >&2
    exit 2
fi
if [[ ! "${CLUSTER_NUMBER}" =~ ^[0-9]+$ ]]; then
    echo "CLUSTER_NUMBER must be a positive integer: ${CLUSTER_NUMBER}" >&2
    exit 2
fi
echo "EPOCH: ${EPOCH}"
echo "EXPERIMENT_ID: ${EXPERIMENT_ID}"
echo "EXPERIMENT_PHASE: ${EXPERIMENT_PHASE}"
echo "ALPHA_BETA_MODE: ${ALPHA_BETA_MODE}"
echo "SCMUG_ALPHA_BETA_GRID: ${SCMUG_ALPHA_BETA_GRID}"
echo "DMKCN_PROJECTIONS: ${DMKCN_PROJECTIONS}"
echo "DMKCN_EMBEDDING_DIMS: ${DMKCN_EMBEDDING_DIMS}"

DATASET_OUTDIR="${OUTPUT_ROOT}/${DATASET}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}_task${SLURM_ARRAY_TASK_ID:-0}"
K_LABEL="$(printf 'k%02d' "${CLUSTER_NUMBER}")"
RUN_STEM="${EXPERIMENT_ID}_${K_LABEL}_${RUN_ID}"
RESULTS_DIR="${DATASET_OUTDIR}/results"
COMPUTATION_DIR="${DATASET_OUTDIR}/computation"
ARTIFACTS_DIR="${DATASET_OUTDIR}/artifacts"
RUN_LOG_DIR="${PROJECT_DIR}/logs/${DATASET}"
mkdir -p "${RESULTS_DIR}" "${COMPUTATION_DIR}" "${ARTIFACTS_DIR}" "${RUN_LOG_DIR}"

RUN_STARTED_JSON="${ARTIFACTS_DIR}/${RUN_STEM}_manifest.json"
RUN_EVENTS_JSONL="${COMPUTATION_DIR}/${RUN_STEM}_events.jsonl"
COMPUTATION_OUTFILE="${COMPUTATION_DIR}/${RUN_STEM}_timings.csv"
RUN_STDOUT="${RUN_LOG_DIR}/${RUN_STEM}.out"
RUN_STDERR="${RUN_LOG_DIR}/${RUN_STEM}.err"
export RUN_ID RUN_STEM RUN_LOG_DIR RUN_STARTED_JSON RUN_EVENTS_JSONL
export COMPUTATION_OUTFILE

exec > >(tee -a "${RUN_STDOUT}") 2> >(tee -a "${RUN_STDERR}" >&2)

RAW_AE="${ARTIFACTS_DIR}/${RUN_STEM}_ae_raw.txt"
RAW_DMKCN="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_raw.txt"
LATENTS_AE="${ARTIFACTS_DIR}/${RUN_STEM}_ae_latents.joblib"
LATENTS_DMKCN="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_latents.joblib"
DMKCN_MANIFEST="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_manifest.json"
OUTFILE="${RESULTS_DIR}/${RUN_STEM}_ablation.tsv"
SUMMARY_OUTFILE="${RESULTS_DIR}/${RUN_STEM}_comparison_summary.csv"
SIGNIFICANCE_OUTFILE="${RESULTS_DIR}/${RUN_STEM}_significance.csv"
VARIANT_OUTFILE="${RESULTS_DIR}/${RUN_STEM}_projection_ablation.tsv"
CD_VALIDATION_OUTFILE="${RESULTS_DIR}/${RUN_STEM}_cd_validation.tsv"
CANONICAL_METRICS_PREFIX="${RUN_STEM}_canonical"
ALL_ALPHA_BETA_METRICS_PREFIX="${RUN_STEM}_all_alpha_beta"

export DATASET_OUTDIR RESULTS_DIR COMPUTATION_DIR ARTIFACTS_DIR
export RAW_AE RAW_DMKCN LATENTS_AE LATENTS_DMKCN DMKCN_MANIFEST
export OUTFILE SUMMARY_OUTFILE SIGNIFICANCE_OUTFILE
export VARIANT_OUTFILE CD_VALIDATION_OUTFILE OUTPUT_ROOT
export CANONICAL_METRICS_PREFIX ALL_ALPHA_BETA_METRICS_PREFIX
export SEEDS REPEAT KMEANS_TIMES EPOCH
export EXPERIMENT_ID EXPERIMENT_PHASE CLUSTER_NUMBER PUBLISHED_BEST_K GROUND_TRUTH_K
export CONFIG_INDEX K_SWEEP K_MIN K_MAX GFM_CORRELATION_RULE
export ALPHA_BETA_MODE SCMUG_ALPHA_BETA_GRID
export RUN_ABLATIONS RUN_ALL_ALPHA_BETA_SUMMARY RUN_STATISTICS
export DMKCN_PROJECTIONS DMKCN_EMBEDDING_DIMS DMKCN_PRIMARY_PROJECTION
export DMKCN_PRIMARY_D DMKCN_GRAPH_NEIGHBORS DMKCN_EVALUATE_EXITS
export DMKCN_EVALUATE_VARIANTS DMKCN_ALLOW_PSEUDO_COUNTS DMKCN_ADAPTER_PILOT
export CD_VALIDATION_ALPHA_BETA_GRID

python3 - <<'PY'
import json
import os
import platform
import socket
import sys
import time


payload = {
    "event": "run_started",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": sys.version,
    "cwd": os.getcwd(),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    },
    "experiment_id": os.environ.get("EXPERIMENT_ID"),
    "run_id": os.environ.get("RUN_ID"),
    "dataset": os.environ.get("DATASET"),
    "k": int(os.environ["CLUSTER_NUMBER"]),
    "published_best_k": int(os.environ["PUBLISHED_BEST_K"]),
    "ground_truth_k": int(os.environ["GROUND_TRUTH_K"]),
    "experiment_phase": os.environ.get("EXPERIMENT_PHASE"),
    "config_index": int(os.environ["CONFIG_INDEX"]),
    "k_sweep": os.environ.get("K_SWEEP") == "true",
    "k_sweep_task_id": os.environ.get("K_SWEEP_TASK_ID"),
    "k_sweep_task_count": os.environ.get("K_SWEEP_TASK_COUNT"),
    "parameters": {
        "cluster_number": os.environ.get("CLUSTER_NUMBER"),
        "n_gfm": os.environ.get("N_GFM"),
        "gfm_correlation_rule": os.environ.get("GFM_CORRELATION_RULE"),
        "cutoffs": os.environ.get("CUTOFFS"),
        "full_alpha": os.environ.get("FULL_ALPHA"),
        "full_beta": os.environ.get("FULL_BETA"),
        "alpha_beta_mode": os.environ.get("ALPHA_BETA_MODE"),
        "alpha_beta_grid": os.environ.get("SCMUG_ALPHA_BETA_GRID"),
        "n_neighbour": os.environ.get("N_NEIGHBOUR"),
        "red_global": os.environ.get("RED_GLOBAL"),
        "red_local": os.environ.get("RED_LOCAL"),
        "seeds": os.environ.get("SEEDS"),
        "repeat": os.environ.get("REPEAT"),
        "kmeans_times": os.environ.get("KMEANS_TIMES"),
        "epoch": os.environ.get("EPOCH"),
        "dmkcn_projections": os.environ.get("DMKCN_PROJECTIONS"),
        "dmkcn_embedding_dims": os.environ.get("DMKCN_EMBEDDING_DIMS"),
        "dmkcn_primary_projection": os.environ.get("DMKCN_PRIMARY_PROJECTION"),
        "dmkcn_primary_d": os.environ.get("DMKCN_PRIMARY_D"),
        "dmkcn_graph_neighbors": os.environ.get("DMKCN_GRAPH_NEIGHBORS"),
        "dmkcn_evaluate_exits": os.environ.get("DMKCN_EVALUATE_EXITS"),
        "dmkcn_evaluate_variants": os.environ.get("DMKCN_EVALUATE_VARIANTS"),
        "dmkcn_adapter_pilot": os.environ.get("DMKCN_ADAPTER_PILOT"),
        "dmkcn_allow_pseudo_counts": os.environ.get("DMKCN_ALLOW_PSEUDO_COUNTS"),
        "cd_validation_alpha_beta_grid": os.environ.get(
            "CD_VALIDATION_ALPHA_BETA_GRID"
        ),
        "run_ablations": os.environ.get("RUN_ABLATIONS"),
        "run_all_alpha_beta_summary": os.environ.get(
            "RUN_ALL_ALPHA_BETA_SUMMARY"
        ),
        "run_statistics": os.environ.get("RUN_STATISTICS"),
    },
    "paths": {
        "dataset_outdir": os.environ.get("DATASET_OUTDIR"),
        "results_dir": os.environ.get("RESULTS_DIR"),
        "computation_dir": os.environ.get("COMPUTATION_DIR"),
        "computation_timings": os.environ.get("COMPUTATION_OUTFILE"),
        "artifacts_dir": os.environ.get("ARTIFACTS_DIR"),
        "run_log_dir": os.environ.get("RUN_LOG_DIR"),
        "raw_ae": os.environ.get("RAW_AE"),
        "raw_dmkcn": os.environ.get("RAW_DMKCN"),
        "latents_ae": os.environ.get("LATENTS_AE"),
        "latents_dmkcn": os.environ.get("LATENTS_DMKCN"),
        "dmkcn_manifest": os.environ.get("DMKCN_MANIFEST"),
        "ablation": os.environ.get("OUTFILE"),
        "canonical_metrics_prefix": os.environ.get("CANONICAL_METRICS_PREFIX"),
        "all_alpha_beta_metrics_prefix": os.environ.get(
            "ALL_ALPHA_BETA_METRICS_PREFIX"
        ),
        "summary": os.environ.get("SUMMARY_OUTFILE"),
        "significance": os.environ.get("SIGNIFICANCE_OUTFILE"),
        "variant_ablation": os.environ.get("VARIANT_OUTFILE"),
        "cd_validation": os.environ.get("CD_VALIDATION_OUTFILE"),
    },
}

with open(os.environ["RUN_STARTED_JSON"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)

with open(os.environ["RUN_EVENTS_JSONL"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
PY

log_event() {
    local event="$1"
    local status="${2:-0}"

    python3 - "${event}" "${status}" <<'PY' || true
import json
import os
import sys
import time

payload = {
    "event": sys.argv[1],
    "status": int(sys.argv[2]),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "experiment_id": os.environ.get("EXPERIMENT_ID"),
    "run_id": os.environ.get("RUN_ID"),
    "dataset": os.environ.get("DATASET"),
    "k": int(os.environ["CLUSTER_NUMBER"]),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
}

with open(os.environ["RUN_EVENTS_JSONL"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
PY
}

on_exit() {
    local status="$?"
    if [ "${status}" -eq 0 ]; then
        log_event "job_finished" "${status}"
    else
        log_event "job_failed" "${status}"
    fi
}

trap on_exit EXIT

echo "=== Python environment ==="
which python3
python3 --version
python3 -m pip --version

echo "=== CUDA check ==="
nvidia-smi || true
which nvcc || true
nvcc --version || true

echo "=== Dependency check ==="
python3 -m pip check || true
python3 - <<'PY'
import anndata
import h5py
import joblib
import numpy
import pandas
import scanpy
import scipy
import sklearn
import torch
import yaml

print("dependency import check ok")
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("scanpy", scanpy.__version__)
print("torch", torch.__version__)
PY

echo "=== Compile gcorr if needed ==="
GCORR_LOCK="${GCORR_DIR}/.gcorr_build.lock"

(
    flock -x 200

    cd "${GCORR_DIR}"

    if [ ! -f "libgcorr.so" ] || [ "gcorr.cu" -nt "libgcorr.so" ]; then
        echo "Task ${SLURM_ARRAY_TASK_ID}: building libgcorr.so"
        rm -f gcorr libgcorr.so gcorr.so
        nvcc -shared -Xcompiler -fPIC -arch=sm_89 -o libgcorr.so gcorr.cu
        ln -sf libgcorr.so gcorr
    else
        echo "Task ${SLURM_ARRAY_TASK_ID}: libgcorr.so up to date"
        ln -sf libgcorr.so gcorr
    fi
) 200>"${GCORR_LOCK}"

export LD_LIBRARY_PATH="${GCORR_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

echo "=== gcorr dynamic library check ==="
ldd "${GCORR_DIR}/libgcorr.so" || true

echo "=== Test gcorr loading ==="
python3 - <<'PY'
from ctypes import cdll
cdll.LoadLibrary("gcorr")
print("gcorr loaded successfully")
PY

echo "=== Run original scMUG (block B = autoencoder) ==="

python3 scMUG.py \
    --dataset "${DATASET}" \
    --cluster_number "${CLUSTER_NUMBER}" \
    --n_gfm "${N_GFM}" \
    --gfm-correlation-rule "${GFM_CORRELATION_RULE}" \
    --cutoffs "${CUTOFFS}" \
    --seeds "${SEEDS}" \
    --repeat "${REPEAT}" \
    --epoch "${EPOCH}" \
    --kmeans_times "${KMEANS_TIMES}" \
    --n_neighbour "${N_NEIGHBOUR}" \
    --thread-num "${SLURM_CPUS_PER_TASK}" \
    --red_global "${RED_GLOBAL}" \
    --red_local "${RED_LOCAL}" \
    --block-b autoencoder \
    --output-tag ae \
    --experiment-id "${EXPERIMENT_ID}" \
    --run-id "${RUN_ID}" \
    --canonical-alpha "${FULL_ALPHA}" \
    --canonical-beta "${FULL_BETA}" \
    --alpha-beta-grid "${SCMUG_ALPHA_BETA_GRID}" \
    --computation-outfile "${COMPUTATION_OUTFILE}" \
    --output-stem "${RUN_STEM}" \
    --output-dir "${ARTIFACTS_DIR}"



echo "=== Run Plugged scMUG (block B = DMKCN) ==="

DMKCN_OPTIONAL_ARGS=()
if [ "${DMKCN_ALLOW_PSEUDO_COUNTS}" = "true" ]; then
    DMKCN_OPTIONAL_ARGS+=(--dmkcn-allow-pseudo-counts)
fi
if [ "${DMKCN_EVALUATE_EXITS}" = "true" ]; then
    DMKCN_OPTIONAL_ARGS+=(--dmkcn-evaluate-exits)
fi

python3 scMUG.py \
    --dataset "${DATASET}" \
    --cluster_number "${CLUSTER_NUMBER}" \
    --n_gfm "${N_GFM}" \
    --gfm-correlation-rule "${GFM_CORRELATION_RULE}" \
    --cutoffs "${CUTOFFS}" \
    --seeds "${SEEDS}" \
    --repeat "${REPEAT}" \
    --epoch "${EPOCH}" \
    --kmeans_times "${KMEANS_TIMES}" \
    --n_neighbour "${N_NEIGHBOUR}" \
    --thread-num "${SLURM_CPUS_PER_TASK}" \
    --red_global "${RED_GLOBAL}" \
    --red_local "${RED_LOCAL}" \
    --block-b dmkcn \
    --output-tag dmkcn \
    --dmkcn-zinb-on-counts \
    --dmkcn-projections "${DMKCN_PROJECTIONS}" \
    --dmkcn-embedding-dims "${DMKCN_EMBEDDING_DIMS}" \
    --dmkcn-primary-projection "${DMKCN_PRIMARY_PROJECTION}" \
    --dmkcn-primary-d "${DMKCN_PRIMARY_D}" \
    --dmkcn-graph-neighbors "${DMKCN_GRAPH_NEIGHBORS}" \
    "${DMKCN_OPTIONAL_ARGS[@]}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --run-id "${RUN_ID}" \
    --canonical-alpha "${FULL_ALPHA}" \
    --canonical-beta "${FULL_BETA}" \
    --alpha-beta-grid "${SCMUG_ALPHA_BETA_GRID}" \
    --computation-outfile "${COMPUTATION_OUTFILE}" \
    --output-stem "${RUN_STEM}" \
    --output-dir "${ARTIFACTS_DIR}"

if [ "${RUN_ABLATIONS}" = "true" ]; then
    echo "=== Run ablation tests from saved latents ==="

    python3 ablation_block_c_d.py \
        --dataset "${DATASET}" \
        --experiment-id "${EXPERIMENT_ID}" \
        --run-id "${RUN_ID}" \
        --cluster_number "${CLUSTER_NUMBER}" \
        --seeds "${SEEDS}" \
        --repeat "${REPEAT}" \
        --kmeans_times "${KMEANS_TIMES}" \
        --n_neighbour "${N_NEIGHBOUR}" \
        --latents-autoencoder "${LATENTS_AE}" \
        --latents-dmkcn "${LATENTS_DMKCN}" \
        --outfile "${OUTFILE}" \
        --thread-num "${SLURM_CPUS_PER_TASK}" \
        --full-alpha "${FULL_ALPHA}" \
        --full-beta "${FULL_BETA}" \
        --red_global "${RED_GLOBAL}" \
        --red_local "${RED_LOCAL}"
fi

if [ "${RUN_ABLATIONS}" = "true" ] && [ "${DMKCN_EVALUATE_VARIANTS}" = "true" ]; then
    echo "=== Evaluate every saved DMKCN K projection through common C/D ==="
    python3 ablation_block_c_d.py \
        --dataset "${DATASET}" \
        --experiment-id "${EXPERIMENT_ID}" \
        --run-id "${RUN_ID}" \
        --cluster_number "${CLUSTER_NUMBER}" \
        --seeds "${SEEDS}" \
        --repeat "${REPEAT}" \
        --kmeans_times "${KMEANS_TIMES}" \
        --n_neighbour "${N_NEIGHBOUR}" \
        --latents-autoencoder "${LATENTS_AE}" \
        --dmkcn-manifest "${DMKCN_MANIFEST}" \
        --outfile "${VARIANT_OUTFILE}" \
        --thread-num "${SLURM_CPUS_PER_TASK}" \
        --full-alpha "${FULL_ALPHA}" \
        --full-beta "${FULL_BETA}" \
        --red_global "${RED_GLOBAL}" \
        --red_local "${RED_LOCAL}"
fi

if [ -n "${CD_VALIDATION_ALPHA_BETA_GRID}" ]; then
    echo "=== Run pre-registered C/D validation conditions ==="
    echo "Alpha/beta pairs: ${CD_VALIDATION_ALPHA_BETA_GRID}"
    python3 ablation_block_c_d.py \
        --dataset "${DATASET}" \
        --experiment-id "${EXPERIMENT_ID}" \
        --run-id "${RUN_ID}" \
        --cluster_number "${CLUSTER_NUMBER}" \
        --seeds "${SEEDS}" \
        --repeat "${REPEAT}" \
        --kmeans_times "${KMEANS_TIMES}" \
        --n_neighbour "${N_NEIGHBOUR}" \
        --latents-autoencoder "${LATENTS_AE}" \
        --latents-dmkcn "${LATENTS_DMKCN}" \
        --outfile "${CD_VALIDATION_OUTFILE}" \
        --thread-num "${SLURM_CPUS_PER_TASK}" \
        --red_global "${RED_GLOBAL}" \
        --red_local "${RED_LOCAL}" \
        --alpha-beta-grid "${CD_VALIDATION_ALPHA_BETA_GRID}"
fi


echo "=== Generate scMUG-style metric summaries ==="

XP_METRICS_ARGS=(
    --dataset "${DATASET}"
    --cluster-number "${CLUSTER_NUMBER}"
    --experiment-id "${EXPERIMENT_ID}"
    --run-id "${RUN_ID}"
    --full-autoencoder "${RAW_AE}"
    --full-dmkcn "${RAW_DMKCN}"
    --full-alpha "${FULL_ALPHA}"
    --full-beta "${FULL_BETA}"
    --outdir "${RESULTS_DIR}"
    --metadata-outdir "${ARTIFACTS_DIR}"
)
if [ "${RUN_ABLATIONS}" = "true" ]; then
    XP_METRICS_ARGS+=(--ablation "${OUTFILE}")
fi

python3 xp_metrics.py \
    "${XP_METRICS_ARGS[@]}" \
    --output-prefix "${CANONICAL_METRICS_PREFIX}"

if [ "${RUN_ALL_ALPHA_BETA_SUMMARY}" = "true" ]; then
    python3 xp_metrics.py \
        "${XP_METRICS_ARGS[@]}" \
        --include-all-alpha-beta \
        --output-prefix "${ALL_ALPHA_BETA_METRICS_PREFIX}"
fi

if [ "${RUN_STATISTICS}" = "true" ]; then
    echo "=== Generate executive summary ==="

    python3 xp_statistics.py \
        --dataset "${DATASET}" \
        --cluster-number "${CLUSTER_NUMBER}" \
        --experiment-id "${EXPERIMENT_ID}" \
        --run-id "${RUN_ID}" \
        --full-autoencoder "${RAW_AE}" \
        --full-dmkcn "${RAW_DMKCN}" \
        --ablation "${OUTFILE}" \
        --full-alpha "${FULL_ALPHA}" \
        --full-beta "${FULL_BETA}" \
        --outfile "${SUMMARY_OUTFILE}" \
        --significance-outfile "${SIGNIFICANCE_OUTFILE}"
fi

echo "=== Job finished for ${DATASET} at $(date) ==="
