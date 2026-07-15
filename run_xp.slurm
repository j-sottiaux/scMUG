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
#SBATCH --array=2,4%2

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

PROJECT_DIR="/home/julien.sottiaux/scMUG"
GCORR_DIR="${PROJECT_DIR}/gcorr"
CONFIG_FILE="${PROJECT_DIR}/configs/data.yaml"

echo "=== Job started on $(hostname) at $(date) ==="
echo "=== SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} ==="

module purge
module load pytorch/2.0.1/gpu

cd "${PROJECT_DIR}"

echo "=== Load dataset configuration ==="
eval "$(
python3 "${PROJECT_DIR}/read_configs.py" \
  --config "${CONFIG_FILE}" \
  --index "${SLURM_ARRAY_TASK_ID}"
)"

echo "Dataset: ${DATASET}"
echo "Cluster number: ${CLUSTER_NUMBER}"
echo "N_GFM: ${N_GFM}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"

if [ "${DMKCN_ADAPTER_PILOT}" = "true" ]; then
    SEEDS="1111,2222,3333"
    REPEAT=1
    DMKCN_PROJECTIONS="spectral_dense,spectral_knn,svd_raw,svd_l2,svd_zscore"
    DMKCN_EMBEDDING_DIMS="9,16,18,32"
    DMKCN_EVALUATE_EXITS=true
    DMKCN_EVALUATE_VARIANTS=true
    OUTPUT_ROOT="./outputs_adapter_pilot"
fi
echo "EPOCH: ${EPOCH}"
echo "DMKCN_PROJECTIONS: ${DMKCN_PROJECTIONS}"
echo "DMKCN_EMBEDDING_DIMS: ${DMKCN_EMBEDDING_DIMS}"

DATASET_OUTDIR="${OUTPUT_ROOT}/${DATASET}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}_task${SLURM_ARRAY_TASK_ID:-0}"
RUN_LOG_DIR="${DATASET_OUTDIR}/run_logs/${RUN_ID}"
mkdir -p "${DATASET_OUTDIR}" "${RUN_LOG_DIR}"

RUN_STARTED_JSON="${RUN_LOG_DIR}/run_started.json"
RUN_EVENTS_JSONL="${RUN_LOG_DIR}/provenance.jsonl"
export RUN_ID RUN_LOG_DIR RUN_STARTED_JSON RUN_EVENTS_JSONL

exec > >(tee -a "${RUN_LOG_DIR}/stdout.log") 2> >(tee -a "${RUN_LOG_DIR}/stderr.log" >&2)

LATENTS_AE="${DATASET_OUTDIR}/latents_ae.joblib"
LATENTS_DMKCN="${DATASET_OUTDIR}/latents_dmkcn.joblib"
OUTFILE="${DATASET_OUTDIR}/ablation_tests.tsv"
SUMMARY_OUTFILE="${DATASET_OUTDIR}/comparison_summary.csv"
SIGNIFICANCE_OUTFILE="${DATASET_OUTDIR}/significance.csv"
METRICS_OUTDIR="${DATASET_OUTDIR}/metrics"
METRICS_ALL_ALPHA_BETA_OUTDIR="${DATASET_OUTDIR}/metrics_all_alpha_beta"
VARIANT_OUTFILE="${DATASET_OUTDIR}/ablation_projection_variants.tsv"

export DATASET_OUTDIR LATENTS_AE LATENTS_DMKCN OUTFILE SUMMARY_OUTFILE
export SIGNIFICANCE_OUTFILE METRICS_OUTDIR METRICS_ALL_ALPHA_BETA_OUTDIR
export VARIANT_OUTFILE OUTPUT_ROOT
export SEEDS REPEAT KMEANS_TIMES EPOCH
export DMKCN_PROJECTIONS DMKCN_EMBEDDING_DIMS DMKCN_PRIMARY_PROJECTION
export DMKCN_PRIMARY_D DMKCN_GRAPH_NEIGHBORS DMKCN_EVALUATE_EXITS
export DMKCN_EVALUATE_VARIANTS DMKCN_ALLOW_PSEUDO_COUNTS DMKCN_ADAPTER_PILOT

python3 - <<'PY'
import json
import os
import platform
import socket
import subprocess
import sys
import time


def git_value(args):
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


status = git_value(["status", "--short"])
payload = {
    "event": "run_started",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "run_id": os.environ["RUN_ID"],
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": sys.version,
    "cwd": os.getcwd(),
    "git": {
        "commit": git_value(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    },
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    },
    "dataset": os.environ.get("DATASET"),
    "parameters": {
        "cluster_number": os.environ.get("CLUSTER_NUMBER"),
        "n_gfm": os.environ.get("N_GFM"),
        "cutoffs": os.environ.get("CUTOFFS"),
        "full_alpha": os.environ.get("FULL_ALPHA"),
        "full_beta": os.environ.get("FULL_BETA"),
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
    },
    "paths": {
        "dataset_outdir": os.environ.get("DATASET_OUTDIR"),
        "run_log_dir": os.environ.get("RUN_LOG_DIR"),
        "latents_ae": os.environ.get("LATENTS_AE"),
        "latents_dmkcn": os.environ.get("LATENTS_DMKCN"),
        "ablation": os.environ.get("OUTFILE"),
        "metrics": os.environ.get("METRICS_OUTDIR"),
        "metrics_all_alpha_beta": os.environ.get("METRICS_ALL_ALPHA_BETA_OUTDIR"),
        "summary": os.environ.get("SUMMARY_OUTFILE"),
        "significance": os.environ.get("SIGNIFICANCE_OUTFILE"),
        "variant_ablation": os.environ.get("VARIANT_OUTFILE"),
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
    "run_id": os.environ.get("RUN_ID"),
    "dataset": os.environ.get("DATASET"),
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

echo "=== Install/fix Python dependencies if needed ==="
PIP_SETUP_LOCK="${PROJECT_DIR}/.slurm_pip_setup.lock"
PIP_SETUP_STAMP="${PROJECT_DIR}/.slurm_pip_setup.ok"

(
    flock -x 201

    if [ ! -f "${PIP_SETUP_STAMP}" ] || [ "${PROJECT_DIR}/requirements.txt" -nt "${PIP_SETUP_STAMP}" ] || [ "${PROJECT_DIR}/run_xp.slurm" -nt "${PIP_SETUP_STAMP}" ]; then
        echo "Task ${SLURM_ARRAY_TASK_ID}: installing Python dependencies"

        python3 -m pip install --user --upgrade pip setuptools wheel

        python3 -m pip install --user --force-reinstall \
          "numpy==1.26.4" \
          "scipy==1.11.4" \
          "zarr<3"

        python3 -m pip install --user -r requirements.txt

        date > "${PIP_SETUP_STAMP}"
    else
        echo "Task ${SLURM_ARRAY_TASK_ID}: Python dependency setup already done"
    fi
) 201>"${PIP_SETUP_LOCK}"

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
    --output-dir "${DATASET_OUTDIR}"



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
    --output-dir "${DATASET_OUTDIR}"

echo "=== Run ablation tests from saved latents ==="

python3 ablation_block_c_d.py \
    --dataset "${DATASET}" \
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

if [ "${DMKCN_EVALUATE_VARIANTS}" = "true" ]; then
    echo "=== Evaluate every saved DMKCN K projection through common C/D ==="
    python3 ablation_block_c_d.py \
        --dataset "${DATASET}" \
        --cluster_number "${CLUSTER_NUMBER}" \
        --seeds "${SEEDS}" \
        --repeat "${REPEAT}" \
        --kmeans_times "${KMEANS_TIMES}" \
        --n_neighbour "${N_NEIGHBOUR}" \
        --latents-autoencoder "${LATENTS_AE}" \
        --dmkcn-manifest "${DATASET_OUTDIR}/latents_dmkcn_manifest.json" \
        --outfile "${VARIANT_OUTFILE}" \
        --thread-num "${SLURM_CPUS_PER_TASK}" \
        --full-alpha "${FULL_ALPHA}" \
        --full-beta "${FULL_BETA}" \
        --red_global "${RED_GLOBAL}" \
        --red_local "${RED_LOCAL}"
fi


echo "=== Generate scMUG-style metric summaries ==="

python3 xp_metrics.py \
  --dataset "${DATASET}" \
  --full-autoencoder "${DATASET_OUTDIR}/${DATASET}s_ae.txt" \
  --full-dmkcn "${DATASET_OUTDIR}/${DATASET}s_dmkcn.txt" \
  --ablation "${OUTFILE}" \
  --full-alpha "${FULL_ALPHA}" \
  --full-beta "${FULL_BETA}" \
  --outdir "${METRICS_OUTDIR}"

python3 xp_metrics.py \
  --dataset "${DATASET}" \
  --full-autoencoder "${DATASET_OUTDIR}/${DATASET}s_ae.txt" \
  --full-dmkcn "${DATASET_OUTDIR}/${DATASET}s_dmkcn.txt" \
  --ablation "${OUTFILE}" \
  --full-alpha "${FULL_ALPHA}" \
  --full-beta "${FULL_BETA}" \
  --include-all-alpha-beta \
  --outdir "${METRICS_ALL_ALPHA_BETA_OUTDIR}"

echo "=== Generate executive summary ==="

python3 xp_statistics.py \
  --dataset "${DATASET}" \
  --full-autoencoder "${DATASET_OUTDIR}/${DATASET}s_ae.txt" \
  --full-dmkcn "${DATASET_OUTDIR}/${DATASET}s_dmkcn.txt" \
  --ablation "${OUTFILE}" \
  --full-alpha "${FULL_ALPHA}" \
  --full-beta "${FULL_BETA}" \
  --outfile "${SUMMARY_OUTFILE}" \
  --significance-outfile "${SIGNIFICANCE_OUTFILE}"

echo "=== Job finished for ${DATASET} at $(date) ==="
