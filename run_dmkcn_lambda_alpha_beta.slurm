#!/bin/bash
#SBATCH --job-name=dmkcn_c2_ab
#SBATCH --output=logs/dmkcn_c2_ab_%A_%a.out
#SBATCH --error=logs/dmkcn_c2_ab_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-3%2
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Campaign task mapping:
#   0 = Baron
#   1 = Darmanis
#   2 = Li
#   3 = Manno
PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/configs/data.yaml}"
LAMBDA_CONFIG="${LAMBDA_CONFIG:-${PROJECT_DIR}/configs/dmkcn_lambdas_triplets}"
BEST_K_FILE="${BEST_K_FILE:-${PROJECT_DIR}/reporting/dmkcn_lambda_campaign2/k_sweep/best_k_by_nmi.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/dmkcn_lambda_campaign2/alpha_beta}"
EXPERIMENT_ID="${EXPERIMENT_ID:-dmkcn_lambda_campaign2_alpha_beta}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
SCMUG_CPU_PARTITION="${SCMUG_CPU_PARTITION:-}"
DRY_RUN="${DRY_RUN:-false}"
SEEDS="${SEEDS:-1111,2222,3333,4444,5555,6666,7777,8888,9999,10000}"
REPEAT="${REPEAT:-3}"
KMEANS_TIMES="${KMEANS_TIMES:-20}"
ALPHA_BETA_GRID="${ALPHA_BETA_GRID:-0:1,0.001:1,0.01:1,0.1:1,1:1,1:0.1,1:0.01,1:0.001,1:0}"
SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID (0-3)}"

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
    require_cpu_partition
    validate_cpu_python_environment
fi

TASK_ARGS=(
    alpha-beta
    --config "${CONFIG_FILE}"
    --lambda-config "${LAMBDA_CONFIG}"
    --best-k "${BEST_K_FILE}"
    --task-id "${SLURM_ARRAY_TASK_ID}"
)
if [ "${DRY_RUN}" = "true" ] && [ ! -f "${BEST_K_FILE}" ]; then
    echo "Dry-run requires the selected-K table: ${BEST_K_FILE}" >&2
    exit 2
fi
load_shell_exports python3 campaign2_tasks.py "${TASK_ARGS[@]}"
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
RUN_LOG_DIR="${PROJECT_DIR}/logs/dmkcn_lambda_campaign2/alpha_beta/${DATASET}"
GRID_TSV="${RESULTS_DIR}/${RUN_STEM}_grid.tsv"
GRID_SUMMARY_TSV="${RESULTS_DIR}/${RUN_STEM}_grid_summary.tsv"
RESOURCE_USAGE="${COMPUTATION_DIR}/${RUN_STEM}_resource_usage.txt"
RUN_MANIFEST="${ARTIFACTS_DIR}/${RUN_STEM}_campaign_manifest.json"

GRID_COMMAND=(
    python3 ablation_block_c_d.py
    --dataset "${DATASET}"
    --experiment-id "${EXPERIMENT_ID}"
    --run-id "${RUN_ID}"
    --cluster-number "${CLUSTER_NUMBER}"
    --seeds "${SEEDS}"
    --repeat "${REPEAT}"
    --kmeans_times "${KMEANS_TIMES}"
    --n_neighbour "${N_NEIGHBOUR}"
    --thread-num "${SLURM_CPUS_PER_TASK:-16}"
    --red_global "${RED_GLOBAL}"
    --red_local "${RED_LOCAL}"
    --alpha-beta-grid "${ALPHA_BETA_GRID}"
    --grid-d-modes spectral
    --arm dmkcn
    --latents-dmkcn "${LATENT_ARTIFACT}"
    --outfile "${GRID_TSV}"
)

echo "Campaign alpha/beta task ${SLURM_ARRAY_TASK_ID}/${CAMPAIGN_TASK_COUNT}: ${DATASET}, K=${CLUSTER_NUMBER}"
echo "Lambda ${LAMBDA_CONFIG_ID}: (${LAMBDA1}, ${LAMBDA2}, ${LAMBDA3})"
printf 'Command:'
printf ' %q' "${GRID_COMMAND[@]}"
printf '\n'
if [ "${DRY_RUN}" = "true" ]; then
    exit 0
fi

mkdir -p "${RESULTS_DIR}" "${COMPUTATION_DIR}" "${ARTIFACTS_DIR}" "${RUN_LOG_DIR}"
for target in \
    "${GRID_TSV}" \
    "${GRID_SUMMARY_TSV}" \
    "${RESOURCE_USAGE}" \
    "${RUN_MANIFEST}"; do
    if [ -e "${target}" ]; then
        echo "Refusing to overwrite existing campaign result: ${target}" >&2
        exit 2
    fi
done

exec > >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.out") \
     2> >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.err" >&2)

export EXPERIMENT_ID RUN_ID DATASET CLUSTER_NUMBER
export LAMBDA_CONFIG LAMBDA_CONFIG_ID LAMBDA1 LAMBDA2 LAMBDA3
export LATENT_ARTIFACT SOURCE_EXPERIMENT_ID SOURCE_RUN_ID
export ALPHA_BETA_GRID SEEDS REPEAT KMEANS_TIMES
export GRID_TSV GRID_SUMMARY_TSV RESOURCE_USAGE RUN_MANIFEST

python3 - <<'PY'
import json
import os
import time

payload = {
    "schema_version": 1,
    "status": "running",
    "phase": "alpha_beta",
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
    "alpha_beta_grid": os.environ["ALPHA_BETA_GRID"],
    "seeds": [int(value) for value in os.environ["SEEDS"].split(",")],
    "repeat": int(os.environ["REPEAT"]),
    "kmeans_times": int(os.environ["KMEANS_TIMES"]),
    "source_experiment_id": os.environ["SOURCE_EXPERIMENT_ID"],
    "source_run_id": os.environ["SOURCE_RUN_ID"],
    "source_latent_artifact": os.environ["LATENT_ARTIFACT"],
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    },
    "paths": {
        "grid_tsv": os.environ["GRID_TSV"],
        "grid_summary_tsv": os.environ["GRID_SUMMARY_TSV"],
        "resource_usage": os.environ["RESOURCE_USAGE"],
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

python3 measure_command_resources.py \
    --output "${RESOURCE_USAGE}" \
    -- "${GRID_COMMAND[@]}"
test -s "${GRID_TSV}"
test -s "${GRID_SUMMARY_TSV}"
test -s "${RESOURCE_USAGE}"

python3 - <<'PY'
import os
import pandas as pd

table = pd.read_csv(os.environ["GRID_TSV"], sep="\t")
expected = 9 * 10 * 3
if len(table) != expected:
    raise ValueError(f"Expected {expected} alpha/beta rows, got {len(table)}.")
PY
