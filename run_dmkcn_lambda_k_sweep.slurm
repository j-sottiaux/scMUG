#!/bin/bash
#SBATCH --job-name=dmkcn_c2_k
#SBATCH --output=logs/dmkcn_c2_k_%A_%a.out
#SBATCH --error=logs/dmkcn_c2_k_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=L40S:1
#SBATCH --partition=gpuai
#SBATCH --array=0-31%2
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Campaign task mapping:
#   0..15  = Li,    K=5..20
#   16..31 = Manno, K=5..20
PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/configs/data.yaml}"
LAMBDA_CONFIG="${LAMBDA_CONFIG:-${PROJECT_DIR}/configs/dmkcn_lambdas_triplets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/dmkcn_lambda_campaign2/k_sweep}"
EXPERIMENT_ID="${EXPERIMENT_ID:-dmkcn_lambda_campaign2_ksweep}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
GCORR_DIR="${GCORR_DIR:-${PROJECT_DIR}/gcorr}"
DRY_RUN="${DRY_RUN:-false}"
SEEDS="${SEEDS:-1111,2222,3333,4444,5555,6666,7777,8888,9999,10000}"
REPEAT="${REPEAT:-3}"
KMEANS_TIMES="${KMEANS_TIMES:-20}"
EPOCH="${EPOCH:-50}"
DMKCN_PROJECTIONS="${DMKCN_PROJECTIONS:-spectral_dense}"
DMKCN_EMBEDDING_DIMS="${DMKCN_EMBEDDING_DIMS:-32}"
DMKCN_PRIMARY_PROJECTION="${DMKCN_PRIMARY_PROJECTION:-spectral_dense}"
DMKCN_PRIMARY_D="${DMKCN_PRIMARY_D:-32}"
DMKCN_GRAPH_NEIGHBORS="${DMKCN_GRAPH_NEIGHBORS:-30}"
SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID (0-31)}"

case "${DRY_RUN}" in
    true|false) ;;
    *)
        echo "DRY_RUN must be true or false, got ${DRY_RUN}." >&2
        exit 2
        ;;
esac

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
if [ "${DRY_RUN}" != "true" ]; then
    module purge
    module load "${SCMUG_PYTHON_MODULE}"
fi

load_shell_exports python3 campaign2_tasks.py k-sweep \
    --config "${CONFIG_FILE}" \
    --lambda-config "${LAMBDA_CONFIG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
SELECTED_DATASET="${DATASET}"
SELECTED_K="${CLUSTER_NUMBER}"
load_shell_exports python3 read_configs.py \
    --config "${CONFIG_FILE}" \
    --index "${CONFIG_INDEX}"
if [ "${DATASET}" != "${SELECTED_DATASET}" ]; then
    echo "Dataset mismatch between campaign and data configurations." >&2
    exit 2
fi
CLUSTER_NUMBER="${SELECTED_K}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export LC_ALL=C

RUN_ID="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}_task${SLURM_ARRAY_TASK_ID}"
K_LABEL="$(printf 'k%02d' "${CLUSTER_NUMBER}")"
RUN_STEM="${EXPERIMENT_ID}_${LAMBDA_CONFIG_ID}_${K_LABEL}_${RUN_ID}"
DATASET_OUTDIR="${OUTPUT_ROOT}/${DATASET}"
RESULTS_DIR="${DATASET_OUTDIR}/results"
COMPUTATION_DIR="${DATASET_OUTDIR}/computation"
ARTIFACTS_DIR="${DATASET_OUTDIR}/artifacts"
RUN_LOG_DIR="${PROJECT_DIR}/logs/dmkcn_lambda_campaign2/k_sweep/${DATASET}"
RAW_DMKCN="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_raw.txt"
LATENTS_DMKCN="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_latents.joblib"
DMKCN_MANIFEST="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_manifest.json"
COMPUTATION_OUTFILE="${COMPUTATION_DIR}/${RUN_STEM}_timings.csv"
RESOURCE_USAGE="${COMPUTATION_DIR}/${RUN_STEM}_resource_usage.txt"
METRICS_PREFIX="${RUN_STEM}_canonical"
METRICS_RAW="${RESULTS_DIR}/${METRICS_PREFIX}_raw_metrics.csv"
METRICS_SEEDS="${RESULTS_DIR}/${METRICS_PREFIX}_seed_means.csv"
METRICS_SUMMARY="${RESULTS_DIR}/${METRICS_PREFIX}_condition_summary.csv"
RUN_MANIFEST="${ARTIFACTS_DIR}/${RUN_STEM}_campaign_manifest.json"
SCMUG_ALPHA_BETA_GRID="${FULL_ALPHA}:${FULL_BETA}"

SCMUG_COMMAND=(
    python3 scMUG.py
    --dataset "${DATASET}"
    --cluster_number "${CLUSTER_NUMBER}"
    --n_gfm "${N_GFM}"
    --gfm-correlation-rule "${GFM_CORRELATION_RULE}"
    --cutoffs "${CUTOFFS}"
    --seeds "${SEEDS}"
    --repeat "${REPEAT}"
    --epoch "${EPOCH}"
    --kmeans_times "${KMEANS_TIMES}"
    --n_neighbour "${N_NEIGHBOUR}"
    --thread-num "${SLURM_CPUS_PER_TASK:-16}"
    --red_global "${RED_GLOBAL}"
    --red_local "${RED_LOCAL}"
    --block-b dmkcn
    --output-tag dmkcn
    --dmkcn-zinb-on-counts
    --dmkcn-projections "${DMKCN_PROJECTIONS}"
    --dmkcn-embedding-dims "${DMKCN_EMBEDDING_DIMS}"
    --dmkcn-primary-projection "${DMKCN_PRIMARY_PROJECTION}"
    --dmkcn-primary-d "${DMKCN_PRIMARY_D}"
    --dmkcn-graph-neighbors "${DMKCN_GRAPH_NEIGHBORS}"
    --dmkcn-lambda-config "${LAMBDA_CONFIG}"
    --experiment-id "${EXPERIMENT_ID}"
    --run-id "${RUN_ID}"
    --canonical-alpha "${FULL_ALPHA}"
    --canonical-beta "${FULL_BETA}"
    --alpha-beta-grid "${SCMUG_ALPHA_BETA_GRID}"
    --computation-outfile "${COMPUTATION_OUTFILE}"
    --output-stem "${RUN_STEM}"
    --output-dir "${ARTIFACTS_DIR}"
)
if [ "${DMKCN_ALLOW_PSEUDO_COUNTS}" = "true" ]; then
    SCMUG_COMMAND+=(--dmkcn-allow-pseudo-counts)
fi

echo "Campaign task ${SLURM_ARRAY_TASK_ID}/${CAMPAIGN_TASK_COUNT}: ${DATASET}, K=${CLUSTER_NUMBER}"
echo "Lambda ${LAMBDA_CONFIG_ID}: (${LAMBDA1}, ${LAMBDA2}, ${LAMBDA3})"
printf 'Command:'
printf ' %q' "${SCMUG_COMMAND[@]}"
printf '\n'
if [ "${DRY_RUN}" = "true" ]; then
    exit 0
fi

mkdir -p "${RESULTS_DIR}" "${COMPUTATION_DIR}" "${ARTIFACTS_DIR}" "${RUN_LOG_DIR}"
for target in \
    "${RAW_DMKCN}" \
    "${LATENTS_DMKCN}" \
    "${DMKCN_MANIFEST}" \
    "${COMPUTATION_OUTFILE}" \
    "${RESOURCE_USAGE}" \
    "${METRICS_RAW}" \
    "${METRICS_SEEDS}" \
    "${METRICS_SUMMARY}" \
    "${RUN_MANIFEST}"; do
    if [ -e "${target}" ]; then
        echo "Refusing to overwrite existing campaign result: ${target}" >&2
        exit 2
    fi
done

exec > >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.out") \
     2> >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.err" >&2)

export EXPERIMENT_ID RUN_ID RUN_STEM DATASET CLUSTER_NUMBER
export LAMBDA_CONFIG LAMBDA_CONFIG_ID LAMBDA1 LAMBDA2 LAMBDA3
export SEEDS REPEAT KMEANS_TIMES FULL_ALPHA FULL_BETA
export RAW_DMKCN LATENTS_DMKCN DMKCN_MANIFEST COMPUTATION_OUTFILE
export RESOURCE_USAGE
export METRICS_RAW METRICS_SEEDS METRICS_SUMMARY RUN_MANIFEST

python3 - <<'PY'
import json
import os
import time

payload = {
    "schema_version": 1,
    "status": "running",
    "phase": "k_sweep",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "experiment_id": os.environ["EXPERIMENT_ID"],
    "run_id": os.environ["RUN_ID"],
    "dataset": os.environ["DATASET"],
    "k": int(os.environ["CLUSTER_NUMBER"]),
    "model": "scMUG-DMKCN",
    "lambdas": {
        "lambda_config_id": os.environ["LAMBDA_CONFIG_ID"],
        "lambda1": float(os.environ["LAMBDA1"]),
        "lambda2": float(os.environ["LAMBDA2"]),
        "lambda3": float(os.environ["LAMBDA3"]),
        "source_file": os.environ["LAMBDA_CONFIG"],
    },
    "canonical_alpha": float(os.environ["FULL_ALPHA"]),
    "canonical_beta": float(os.environ["FULL_BETA"]),
    "seeds": [int(value) for value in os.environ["SEEDS"].split(",")],
    "repeat": int(os.environ["REPEAT"]),
    "kmeans_times": int(os.environ["KMEANS_TIMES"]),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    },
    "paths": {
        "raw_dmkcn": os.environ["RAW_DMKCN"],
        "latents_dmkcn": os.environ["LATENTS_DMKCN"],
        "dmkcn_manifest": os.environ["DMKCN_MANIFEST"],
        "computation_timings": os.environ["COMPUTATION_OUTFILE"],
        "resource_usage": os.environ["RESOURCE_USAGE"],
        "metrics_raw": os.environ["METRICS_RAW"],
        "metrics_seed_means": os.environ["METRICS_SEEDS"],
        "metrics_condition_summary": os.environ["METRICS_SUMMARY"],
    },
}
with open(os.environ["RUN_MANIFEST"], "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

on_exit() {
    local status="$?"
    python3 - "${status}" <<'PY' || true
import json
import os
import sys
import time

path = os.environ["RUN_MANIFEST"]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
status = int(sys.argv[1])
payload["status"] = "completed" if status == 0 else "failed"
payload["exit_status"] = status
payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
temporary = path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}
trap on_exit EXIT

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
print("Dependency imports OK; torch", torch.__version__)
PY
nvidia-smi

GCORR_LOCK="${GCORR_DIR}/.gcorr_build.lock"
(
    flock -x 200
    cd "${GCORR_DIR}"
    if [ ! -f "libgcorr.so" ] || [ "gcorr.cu" -nt "libgcorr.so" ]; then
        nvcc -shared -Xcompiler -fPIC -arch=sm_89 \
            -o "libgcorr.so.new" "gcorr.cu"
        mv "libgcorr.so.new" "libgcorr.so"
    fi
    ln -sfn "libgcorr.so" "gcorr"
) 200>"${GCORR_LOCK}"
export LD_LIBRARY_PATH="${GCORR_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
ldd "${GCORR_DIR}/libgcorr.so"

test -x /usr/bin/time
/usr/bin/time -v -o "${RESOURCE_USAGE}" "${SCMUG_COMMAND[@]}"

python3 xp_metrics.py \
    --dataset "${DATASET}" \
    --cluster-number "${CLUSTER_NUMBER}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --run-id "${RUN_ID}" \
    --full-dmkcn "${RAW_DMKCN}" \
    --full-alpha "${FULL_ALPHA}" \
    --full-beta "${FULL_BETA}" \
    --lambda-config-id "${LAMBDA_CONFIG_ID}" \
    --lambda1 "${LAMBDA1}" \
    --lambda2 "${LAMBDA2}" \
    --lambda3 "${LAMBDA3}" \
    --outdir "${RESULTS_DIR}" \
    --metadata-outdir "${ARTIFACTS_DIR}" \
    --output-prefix "${METRICS_PREFIX}"

for required in \
    "${RAW_DMKCN}" \
    "${LATENTS_DMKCN}" \
    "${DMKCN_MANIFEST}" \
    "${COMPUTATION_OUTFILE}" \
    "${RESOURCE_USAGE}" \
    "${METRICS_RAW}" \
    "${METRICS_SEEDS}" \
    "${METRICS_SUMMARY}"; do
    test -s "${required}"
done
