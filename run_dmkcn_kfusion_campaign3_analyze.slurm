#!/bin/bash
#SBATCH --job-name=dmkcn_c3_analyze
#SBATCH --output=logs/dmkcn_kfusion_campaign3/slurm/analyze_%A_%a.out
#SBATCH --error=logs/dmkcn_kfusion_campaign3/slurm/analyze_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-7%2
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
CAMPAIGN_INSTANCE_ID="${CAMPAIGN_INSTANCE_ID:?Set CAMPAIGN_INSTANCE_ID}"
CAMPAIGN_CONFIG="${CAMPAIGN_CONFIG:-${PROJECT_DIR}/configs/dmkcn_kfusion_campaign3.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/dmkcn_kfusion_campaign3/${CAMPAIGN_INSTANCE_ID}}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
SCMUG_CPU_PARTITION="${SCMUG_CPU_PARTITION:?Set SCMUG_CPU_PARTITION}"
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
require_cpu_partition
validate_cpu_python_environment
load_shell_exports python3 campaign3_tasks.py \
    --campaign-config "${CAMPAIGN_CONFIG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export LC_ALL=C
export LD_LIBRARY_PATH="${GCORR_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

RUN_STEM="${CAMPAIGN_ID}_${DATASET}_${REGIME}"
TRAIN_ROOT="${OUTPUT_ROOT}/training/${DATASET}/${REGIME}"
TRAIN_ARTIFACTS="${TRAIN_ROOT}/artifacts"
TRAIN_MANIFEST="${TRAIN_ARTIFACTS}/${RUN_STEM}_training_manifest.json"
KERNELS_DIR="${TRAIN_ARTIFACTS}/${RUN_STEM}_dmkcn_kernels"
LATENTS_DMKCN="${TRAIN_ARTIFACTS}/${RUN_STEM}_dmkcn_latents.joblib"
PREDICTIONS_DMKCN="${TRAIN_ARTIFACTS}/${RUN_STEM}_dmkcn_predictions.joblib"
REFERENCE_METRICS="${TRAIN_ROOT}/results/${RUN_STEM}_current_cd_raw_metrics.csv"
CELL_METADATA="${TRAIN_ARTIFACTS}/${RUN_STEM}_dmkcn_cell_metadata.csv"
GFM_MEMBERSHIP="${TRAIN_ARTIFACTS}/${RUN_STEM}_dmkcn_gfm_membership.json"

ANALYSIS_ROOT="${OUTPUT_ROOT}/analysis/${DATASET}/${REGIME}"
ANALYSIS_PREFIX="${RUN_STEM}"
COMPUTATION_DIR="${ANALYSIS_ROOT}/computation"
RESOURCE_USAGE="${COMPUTATION_DIR}/${RUN_STEM}_resource_usage.txt"
RUN_LOG_DIR="${PROJECT_DIR}/logs/dmkcn_kfusion_campaign3/${CAMPAIGN_INSTANCE_ID}/analysis/${DATASET}/${REGIME}"

python3 - "${TRAIN_MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("status") != "completed" or payload.get("exit_status") != 0:
    raise RuntimeError(f"Training manifest is not complete: {sys.argv[1]}")
PY

if [ -e "${ANALYSIS_ROOT}" ]; then
    echo "Refusing to overwrite existing campaign-3 analysis root: ${ANALYSIS_ROOT}" >&2
    exit 2
fi
mkdir -p "${COMPUTATION_DIR}" "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.out") \
     2> >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.err" >&2)

ANALYSIS_COMMAND=(
    python3 analyze_dmkcn_kfusion_campaign3.py
    --campaign-id "${CAMPAIGN_ID}"
    --campaign-instance-id "${CAMPAIGN_INSTANCE_ID}"
    --dataset "${DATASET}"
    --regime "${REGIME}"
    --regime-role "${REGIME_ROLE}"
    --k-train "${K_TRAIN}"
    --k-cluster "${K_CLUSTER}"
    --alpha "${ALPHA}"
    --beta "${BETA}"
    --lambda-config-id "${LAMBDA_CONFIG_ID}"
    --lambda1 "${LAMBDA1}"
    --lambda2 "${LAMBDA2}"
    --lambda3 "${LAMBDA3}"
    --n-gfm "${N_GFM}"
    --projection-dimension "${DMKCN_PROJECTION_DIMENSION}"
    --seeds "${SEEDS}"
    --repeats "${REPEAT}"
    --kmeans-times "${KMEANS_TIMES}"
    --n-neighbour "${N_NEIGHBOUR}"
    --red-global "${RED_GLOBAL}"
    --red-local "${RED_LOCAL}"
    --thread-num "${SLURM_CPUS_PER_TASK:-16}"
    --kernels-dir "${KERNELS_DIR}"
    --latents "${LATENTS_DMKCN}"
    --reference-predictions "${PREDICTIONS_DMKCN}"
    --reference-metrics "${REFERENCE_METRICS}"
    --cell-metadata "${CELL_METADATA}"
    --gfm-membership "${GFM_MEMBERSHIP}"
    --output-root "${ANALYSIS_ROOT}"
    --output-prefix "${ANALYSIS_PREFIX}"
)

printf 'Command:'
printf ' %q' "${ANALYSIS_COMMAND[@]}"
printf '\n'
python3 measure_command_resources.py \
    --output "${RESOURCE_USAGE}" \
    -- "${ANALYSIS_COMMAND[@]}"

for required in \
    "${ANALYSIS_ROOT}/results/${ANALYSIS_PREFIX}_raw_metrics.csv" \
    "${ANALYSIS_ROOT}/results/${ANALYSIS_PREFIX}_seed_means.csv" \
    "${ANALYSIS_ROOT}/results/${ANALYSIS_PREFIX}_condition_summary.csv" \
    "${ANALYSIS_ROOT}/results/${ANALYSIS_PREFIX}_paired_contrasts.csv" \
    "${ANALYSIS_ROOT}/artifacts/${ANALYSIS_PREFIX}_predictions.npz" \
    "${ANALYSIS_ROOT}/artifacts/${ANALYSIS_PREFIX}_analysis_manifest.json" \
    "${ANALYSIS_ROOT}/diagnostics/${ANALYSIS_PREFIX}_integrity_report.json" \
    "${RESOURCE_USAGE}"; do
    test -s "${required}"
done
