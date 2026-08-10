#!/bin/bash
#SBATCH --job-name=dmkcn_c3_train
#SBATCH --output=logs/dmkcn_kfusion_campaign3/slurm/train_%A_%a.out
#SBATCH --error=logs/dmkcn_kfusion_campaign3/slurm/train_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=L40S:1
#SBATCH --partition=gpuai
#SBATCH --array=0-7%2
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
CAMPAIGN_INSTANCE_ID="${CAMPAIGN_INSTANCE_ID:?Set CAMPAIGN_INSTANCE_ID}"
CAMPAIGN_CONFIG="${CAMPAIGN_CONFIG:-${PROJECT_DIR}/configs/dmkcn_kfusion_campaign3.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/dmkcn_kfusion_campaign3/${CAMPAIGN_INSTANCE_ID}}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
GCORR_DIR="${GCORR_DIR:-${PROJECT_DIR}/gcorr}"
SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID (0-7)}"
if [[ ! "${CAMPAIGN_INSTANCE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Invalid CAMPAIGN_INSTANCE_ID: ${CAMPAIGN_INSTANCE_ID}" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
module purge
module load "${SCMUG_PYTHON_MODULE}"
load_shell_exports python3 campaign3_tasks.py \
    --campaign-config "${CAMPAIGN_CONFIG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export LC_ALL=C

RUN_ID="${CAMPAIGN_INSTANCE_ID}_job${SLURM_JOB_ID:-local}_task${SLURM_ARRAY_TASK_ID}"
RUN_STEM="${CAMPAIGN_ID}_${DATASET}_${REGIME}"
TASK_ROOT="${OUTPUT_ROOT}/training/${DATASET}/${REGIME}"
RESULTS_DIR="${TASK_ROOT}/results"
COMPUTATION_DIR="${TASK_ROOT}/computation"
ARTIFACTS_DIR="${TASK_ROOT}/artifacts"
RUN_LOG_DIR="${PROJECT_DIR}/logs/dmkcn_kfusion_campaign3/${CAMPAIGN_INSTANCE_ID}/training/${DATASET}/${REGIME}"

RAW_DMKCN="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_raw.txt"
LATENTS_DMKCN="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_latents.joblib"
PREDICTIONS_DMKCN="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_predictions.joblib"
DMKCN_MANIFEST="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_manifest.json"
DIAGNOSTICS_JSONL="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_diagnostics.jsonl"
KERNELS_DIR="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_kernels"
CELL_METADATA="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_cell_metadata.csv"
GFM_MEMBERSHIP="${ARTIFACTS_DIR}/${RUN_STEM}_dmkcn_gfm_membership.json"
COMPUTATION_OUTFILE="${COMPUTATION_DIR}/${RUN_STEM}_timings.csv"
RESOURCE_USAGE="${COMPUTATION_DIR}/${RUN_STEM}_resource_usage.txt"
METRICS_PREFIX="${RUN_STEM}_current_cd"
METRICS_RAW="${RESULTS_DIR}/${METRICS_PREFIX}_raw_metrics.csv"
METRICS_SEEDS="${RESULTS_DIR}/${METRICS_PREFIX}_seed_means.csv"
METRICS_SUMMARY="${RESULTS_DIR}/${METRICS_PREFIX}_condition_summary.csv"
TASK_MANIFEST="${ARTIFACTS_DIR}/${RUN_STEM}_training_manifest.json"

if [ -e "${TASK_ROOT}" ]; then
    echo "Refusing to overwrite existing campaign-3 task root: ${TASK_ROOT}" >&2
    exit 2
fi
mkdir -p "${RESULTS_DIR}" "${COMPUTATION_DIR}" "${ARTIFACTS_DIR}" "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.out") \
     2> >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.err" >&2)

SCMUG_COMMAND=(
    python3 scMUG.py
    --dataset "${DATASET}"
    --cluster_number "${K_TRAIN}"
    --n_gfm "${N_GFM}"
    --gfm-correlation-rule "${GFM_CORRELATION_RULE}"
    --cutoffs "${CUTOFFS}"
    --seeds "${SEEDS}"
    --repeat "${REPEAT}"
    --epoch 50
    --kmeans_times "${KMEANS_TIMES}"
    --n_neighbour "${N_NEIGHBOUR}"
    --thread-num "${SLURM_CPUS_PER_TASK:-16}"
    --red_global "${RED_GLOBAL}"
    --red_local "${RED_LOCAL}"
    --block-b dmkcn
    --output-tag dmkcn
    --dmkcn-zinb-on-counts
    --dmkcn-full-training
    --dmkcn-projections "${DMKCN_PROJECTION}"
    --dmkcn-embedding-dims "${DMKCN_PROJECTION_DIMENSION}"
    --dmkcn-primary-projection "${DMKCN_PROJECTION}"
    --dmkcn-primary-d "${DMKCN_PROJECTION_DIMENSION}"
    --dmkcn-graph-neighbors 30
    --dmkcn-lambda-config "${LAMBDA_CONFIG}"
    --dmkcn-lambda-config-id "${LAMBDA_CONFIG_ID}"
    --dmkcn-export-kernels
    --experiment-id "${CAMPAIGN_ID}"
    --run-id "${RUN_ID}"
    --canonical-alpha "${ALPHA}"
    --canonical-beta "${BETA}"
    --alpha-beta-grid "${ALPHA}:${BETA}"
    --computation-outfile "${COMPUTATION_OUTFILE}"
    --output-stem "${RUN_STEM}"
    --output-dir "${ARTIFACTS_DIR}"
)
if [ "${DMKCN_ALLOW_PSEUDO_COUNTS}" = "true" ]; then
    SCMUG_COMMAND+=(--dmkcn-allow-pseudo-counts)
fi

export CAMPAIGN_INSTANCE_ID RUN_ID RUN_STEM TASK_MANIFEST
export SCMUG_PYTHON_MODULE
export DATASET REGIME REGIME_ROLE K_TRAIN K_CLUSTER ALPHA BETA
export LAMBDA_CONFIG LAMBDA_CONFIG_ID LAMBDA1 LAMBDA2 LAMBDA3
export LAMBDA_SOURCE_CAMPAIGN_ID
export SEEDS REPEAT N_GFM RAW_DMKCN LATENTS_DMKCN PREDICTIONS_DMKCN
export DMKCN_MANIFEST KERNELS_DIR CELL_METADATA GFM_MEMBERSHIP
export COMPUTATION_OUTFILE RESOURCE_USAGE METRICS_RAW METRICS_SEEDS METRICS_SUMMARY

python3 - <<'PY'
import json
import os
import platform
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

payload = {
    "schema_version": 1,
    "status": "running",
    "phase": "training",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "campaign_instance_id": os.environ["CAMPAIGN_INSTANCE_ID"],
    "run_id": os.environ["RUN_ID"],
    "dataset": os.environ["DATASET"],
    "regime": os.environ["REGIME"],
    "regime_role": os.environ["REGIME_ROLE"],
    "k_train": int(os.environ["K_TRAIN"]),
    "k_cluster": int(os.environ["K_CLUSTER"]),
    "alpha": float(os.environ["ALPHA"]),
    "beta": float(os.environ["BETA"]),
    "lambdas": {
        "lambda_config_id": os.environ["LAMBDA_CONFIG_ID"],
        "lambda1": float(os.environ["LAMBDA1"]),
        "lambda2": float(os.environ["LAMBDA2"]),
        "lambda3": float(os.environ["LAMBDA3"]),
        "source_file": os.environ["LAMBDA_CONFIG"],
        "source_campaign_id": os.environ["LAMBDA_SOURCE_CAMPAIGN_ID"],
    },
    "seeds": [int(value) for value in os.environ["SEEDS"].split(",")],
    "repeat": int(os.environ["REPEAT"]),
    "n_gfm": int(os.environ["N_GFM"]),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    },
    "source_control": {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status_porcelain": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines(),
    },
    "software": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_module": os.environ.get("SCMUG_PYTHON_MODULE"),
        "packages": {
            name: package_version(name)
            for name in (
                "anndata", "h5py", "joblib", "numpy", "pandas", "scanpy",
                "scipy", "scikit-learn", "torch", "umap-learn", "PyYAML"
            )
        },
    },
    "paths": {
        name.lower(): os.environ[name]
        for name in (
            "RAW_DMKCN", "LATENTS_DMKCN", "PREDICTIONS_DMKCN",
            "DMKCN_MANIFEST", "KERNELS_DIR", "CELL_METADATA",
            "GFM_MEMBERSHIP", "COMPUTATION_OUTFILE", "RESOURCE_USAGE",
            "METRICS_RAW", "METRICS_SEEDS", "METRICS_SUMMARY"
        )
    },
}
with open(os.environ["TASK_MANIFEST"], "x", encoding="utf-8") as handle:
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

path = os.environ["TASK_MANIFEST"]
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

printf 'Command:'
printf ' %q' "${SCMUG_COMMAND[@]}"
printf '\n'
python3 measure_command_resources.py \
    --output "${RESOURCE_USAGE}" \
    -- "${SCMUG_COMMAND[@]}"

python3 xp_metrics.py \
    --dataset "${DATASET}" \
    --cluster-number "${K_CLUSTER}" \
    --experiment-id "${CAMPAIGN_ID}" \
    --run-id "${RUN_ID}" \
    --full-dmkcn "${RAW_DMKCN}" \
    --full-alpha "${ALPHA}" \
    --full-beta "${BETA}" \
    --lambda-config-id "${LAMBDA_CONFIG_ID}" \
    --lambda1 "${LAMBDA1}" \
    --lambda2 "${LAMBDA2}" \
    --lambda3 "${LAMBDA3}" \
    --outdir "${RESULTS_DIR}" \
    --metadata-outdir "${ARTIFACTS_DIR}" \
    --output-prefix "${METRICS_PREFIX}"

for required in \
    "${RAW_DMKCN}" "${LATENTS_DMKCN}" "${PREDICTIONS_DMKCN}" \
    "${DMKCN_MANIFEST}" "${DIAGNOSTICS_JSONL}" "${CELL_METADATA}" \
    "${GFM_MEMBERSHIP}" "${COMPUTATION_OUTFILE}" "${RESOURCE_USAGE}" \
    "${METRICS_RAW}" "${METRICS_SEEDS}" "${METRICS_SUMMARY}"; do
    test -s "${required}"
done
test "$(find "${KERNELS_DIR}" -maxdepth 1 -type f -name 'seed*_kernel_representations.npz' | wc -l)" -eq 10
